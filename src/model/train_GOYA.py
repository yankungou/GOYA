import time
import glob
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import misc
from misc import NativeScalerWithGradNormCount as NativeScaler

from dataset import GOYADataset
from utils import AverageMeter, ProgressMeter
from get_contrastive_pairs import get_cs_sep_indices_tuples
from net import make_GOYA

    
def resume_epoch_model_ddp(args, model, optimizer):
    '''put the model path/dir to args.resume if needed'''
    best_loss = 100000
    best_content_loss = 100000    
    best_style_loss = 100000

    args.start_epoch = 0
    best_batch_idx = 0
    best_content_batch_idx = 0
    best_style_batch_idx = 0

    if args.resume:
        if os.path.isdir(args.resume) or os.path.isfile(args.resume):
            print("loading model")
            if os.path.isdir(args.resume):
                model_path = sorted(glob.glob(args.resume + '/*'), key=os.path.getmtime)[-1]
            elif os.path.isfile(args.resume):
                model_path = args.resume
            
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            args.start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            best_content_loss = checkpoint['best_content_loss']
            best_style_loss = checkpoint['best_style_loss']

            best_batch_idx = checkpoint['best_batch_idx']
            best_content_batch_idx = checkpoint['best_content_batch_idx']
            best_style_batch_idx = checkpoint['best_style_batch_idx']

            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
        else:
            raise OSError("no checkpoint found at '{}'".format(args.resume))
    else:
        print("Not loading any checkpoint")

    return best_loss, best_content_loss, best_style_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx, model, optimizer


def save_best_state_model(args, epoch, model_without_ddp, best_loss, best_content_loss, best_style_loss, optimizer, val_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx, best_type):
    '''save model when it is best in content or style. 
    best_type: content, style
    '''
    state_model = {
        'epoch': epoch,
        'state_dict': model_without_ddp.state_dict(),
        'best_loss': best_loss,
        'best_content_loss': best_content_loss,
        'best_style_loss': best_style_loss,
        'optimizer': optimizer.state_dict(),
        'curr_val': val_loss,
        'best_batch_idx': best_batch_idx,
        'best_content_batch_idx': best_content_batch_idx,
        'best_style_batch_idx': best_style_batch_idx,
    }
    
    os.makedirs(os.path.join(args.model_root, args.model), exist_ok=True)
    save_model_path = os.path.join(args.model_root, args.model, f'{args.model}_{best_type}.pth.tar')

    torch.save(state_model, save_model_path)
    print(f'Best {best_type} model saved to {save_model_path}')
    

def train_model_epoch(args, train_loader, val_loader, model, model_without_ddp, optimizer, epoch, device, best_loss, best_content_loss, best_style_loss, main_loss, cls_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx, loss_scaler, distance_matrix):
    model.train()
    losses = AverageMeter('Total loss', ':.4e')

    content_contra_losses = AverageMeter('Content contrastive loss', ':.4e')
    style_contra_losses = AverageMeter('Style contrastive loss', ':.4e')
    style_clf_losses = AverageMeter('Style classification loss', ':.4e')

    progress = ProgressMeter(len(train_loader), 
                            [losses, content_contra_losses, style_contra_losses, style_clf_losses], 
                            prefix="Epoch: [{}]".format(epoch))
    
    L = len(train_loader)
    accum_iter = args.accum_iter
    batch_val_slot = 50

    with tqdm(postfix=dict, mininterval=0.3) as pbar:
        for batch_idx, (feature_ori, prompt) in enumerate(train_loader):
            content_target, style_target = prompt
            model.train()
            curr_iter = batch_idx + L * epoch
            
            # we use a per iteration (instead of per epoch) lr scheduler
            optimizer.param_groups[0]['lr'] = args.lr * (1 - float(curr_iter) / (args.nepochs * L)) ** args.lr_decay
            batch_loss_value = 0.0
            feature_ori = feature_ori.to(device)
            
            content_target = content_target.to(device, non_blocking=True)
            style_target = style_target.to(device, non_blocking=True)
            
            out_feature_c, out_feature_s, _, _, out_feature_s_clf = model(feature_ori)

            # consider content and style separately
            content_indices_tuple, style_indices_tuple = get_cs_sep_indices_tuples(content_target, style_target, distance_matrix)
        
            content_pos_target = torch.ones(content_indices_tuple[0].size(0), device=device)
            content_neg_target = - torch.ones(content_indices_tuple[2].size(0), device=device)
                            
            style_pos_target = torch.ones(style_indices_tuple[0].size(0), device=device)
            style_neg_target = - torch.ones(style_indices_tuple[2].size(0), device=device)

            content_batch_loss = main_loss(out_feature_c[content_indices_tuple[0]], out_feature_c[content_indices_tuple[1]], content_pos_target) + main_loss(out_feature_c[content_indices_tuple[2]], out_feature_c[content_indices_tuple[3]], content_neg_target)
            style_batch_loss = main_loss(out_feature_s[style_indices_tuple[0]], out_feature_s[style_indices_tuple[1]], style_pos_target) + main_loss(out_feature_s[style_indices_tuple[2]], out_feature_s[style_indices_tuple[3]], style_neg_target)
            
            style_clf_loss = cls_loss(out_feature_s_clf, style_target)
            loss = content_batch_loss + style_batch_loss + style_clf_loss

            # batch_loss_value is for visualization
            batch_loss_value = loss.item()
            
            losses.update(batch_loss_value, feature_ori.size(0))
            content_contra_losses.update(content_batch_loss.item(), feature_ori.size(0))
            style_contra_losses.update(style_batch_loss.item(), feature_ori.size(0))
            style_clf_losses.update(style_clf_loss.item(), feature_ori.size(0))

            torch.cuda.synchronize()
        
            # save and print the status
            if misc.is_main_process():
                log_stats = {**{'epoch': epoch,
                                'batch': batch_idx,
                                'train batch loss': '%0.4f' % batch_loss_value,
                                'train batch content_batch_loss': '%0.4f' % content_batch_loss.item(),
                                'train batch style_batch_loss': '%0.4f' % style_batch_loss.item(),
                                'train batch style_clf_loss': '%0.4f' % style_clf_loss.item(),
                                'learning rate': optimizer.param_groups[0]['lr'],
                            }}
                pbar.set_postfix(log_stats)
        
            # update loss, optimizer
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(curr_iter + 1) % accum_iter == 0)
            if (curr_iter + 1) % accum_iter == 0:
                optimizer.zero_grad()
            
            if misc.is_main_process():
                if curr_iter % batch_val_slot == 0:
                    val_loss, val_content_batch_loss, val_style_batch_loss, val_style_clf_loss = val_model(args, val_loader, model, epoch, device, main_loss, cls_loss, distance_matrix)

                    # Save if it is the best model
                    is_content_best = val_content_batch_loss < best_content_loss
                    best_content_loss = min(val_content_batch_loss, best_content_loss)
                    if is_content_best:
                        best_content_batch_idx = curr_iter
                        save_best_state_model(args, epoch, model_without_ddp, best_loss, best_content_loss, best_style_loss, optimizer, val_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx, best_type='content')

                    val_style_batch_loss = val_style_batch_loss + val_style_clf_loss
                    is_style_best = val_style_batch_loss < best_style_loss
                    best_style_loss = min(val_style_batch_loss, best_style_loss)
                    if is_style_best:
                        best_style_batch_idx = curr_iter
                        save_best_state_model(args, epoch, model_without_ddp, best_loss, best_content_loss, best_style_loss, optimizer, val_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx, best_type='style')

                    print(f'** Validation: {best_content_loss} (best content loss) -- {val_content_batch_loss} (current content loss) / {best_style_loss} (best style loss) -- {val_style_batch_loss} (current style loss)')
        
        if misc.is_main_process():
            epoch_log_stats = {**{'epoch': epoch,
                                'train loss': '%0.4f' % losses.avg, 
                                'train epoch content contrastive loss': '%0.4f' % content_contra_losses.avg, 
                                'train epoch style contrastive loss': '%0.4f' % style_contra_losses.avg, 
                                'train epoch style classification loss': '%0.4f' % style_clf_losses.avg, 
                                }}
            pbar.set_postfix(epoch_log_stats)
            pbar.update(1)
            
            progress.display(curr_iter + 1)

    return best_loss, best_content_loss, best_style_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx


def val_model(args, val_loader, model, epoch, device, main_loss, cls_loss, distance_matrix):
    model.eval()
    val_losses = AverageMeter('Total val loss', ':.4e')

    content_contra_losses = AverageMeter('Content contrastive loss', ':.4e')
    style_contra_losses = AverageMeter('Style contrastive loss', ':.4e')
    style_clf_losses = AverageMeter('Style classification loss', ':.4e')

    val_progress = ProgressMeter(len(val_loader), 
                            [val_losses, content_contra_losses, style_contra_losses, style_clf_losses], 
                            prefix="Epoch: [{}]".format(epoch))
    
    with tqdm(postfix=dict, mininterval=0.3) as pbar:
        with torch.no_grad():
            for _, (feature_ori, prompt) in enumerate(val_loader):
                content_target, style_target = prompt

                feature_ori = feature_ori.to(device)

                content_target = content_target.to(device, non_blocking=True)
                style_target = style_target.to(device, non_blocking=True)
                out_feature_c, out_feature_s, _, _, out_feature_s_clf = model(feature_ori)

                # consider content and style separately
                content_indices_tuple, style_indices_tuple = get_cs_sep_indices_tuples(content_target, style_target, distance_matrix)
                
                content_pos_target = torch.ones(content_indices_tuple[0].size(0), device=device)
                content_neg_target = - torch.ones(content_indices_tuple[2].size(0), device=device)
                                
                style_pos_target = torch.ones(style_indices_tuple[0].size(0), device=device)
                style_neg_target = - torch.ones(style_indices_tuple[2].size(0), device=device)

                content_batch_loss = main_loss(out_feature_c[content_indices_tuple[0]], out_feature_c[content_indices_tuple[1]], content_pos_target) + main_loss(out_feature_c[content_indices_tuple[2]], out_feature_c[content_indices_tuple[3]], content_neg_target)
                style_batch_loss = main_loss(out_feature_s[style_indices_tuple[0]], out_feature_s[style_indices_tuple[1]], style_pos_target) + main_loss(out_feature_s[style_indices_tuple[2]], out_feature_s[style_indices_tuple[3]], style_neg_target)
                
                style_clf_loss = cls_loss(out_feature_s_clf, style_target)
                loss = content_batch_loss + style_batch_loss + style_clf_loss

                # batch_loss_value is for visualization
                batch_loss_value = loss.item()
                
                val_losses.update(batch_loss_value, feature_ori.size(0))
                content_contra_losses.update(content_batch_loss.item(), feature_ori.size(0))
                style_contra_losses.update(style_batch_loss.item(), feature_ori.size(0))
                style_clf_losses.update(style_clf_loss.item(), feature_ori.size(0))
                
        epoch_log_stats_val = {**{'epoch': epoch,
                    'val loss': '%0.4f' % val_losses.avg, 
                    'val epoch content_contra_loss': '%0.4f' % content_contra_losses.avg,
                    'val epoch style_contra_loss': '%0.4f' % style_contra_losses.avg,
                    'val epoch style_clf_loss': '%0.4f' % style_clf_losses.avg,
                    }}
        pbar.set_postfix(epoch_log_stats_val)
        val_progress.display_summary()
        return val_losses.avg, content_contra_losses.avg, style_contra_losses.avg, style_clf_losses.avg
        

def train(args):
        
    # ----- ddp -----
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # cudnn.enabled = False
    
    # ---------
    if args.distributed:
        print("using {} device.".format(device))
    else:
        device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

    model = make_GOYA()
    
    # ------ Prepare data ------- #
    # CLIP feature of generated images
    train_feature_path = os.path.join(args.feature_dir, 'generated_train_clip_feature.pickle')
    val_feature_path = os.path.join(args.feature_dir, 'generated_val_clip_feature.pickle')

    if not os.path.exists(train_feature_path):
        print('Training feature is not found. You can download it from here: https://drive.google.com/file/d/1gc8BcXa8e1f6N2KHvDJfTdKp7cgXdWyX/view?usp=drive_link')
        print(f'Please place it in {args.feature_dir}.')
    if not os.path.exists(val_feature_path):
        print('Validation feature is not found. You can download it from here: https://drive.google.com/file/d/1_eTdgNHDBTNaFYs1J7k-Drn_FVDkDLKe/view?usp=drive_link')
        print(f'Please place it in {args.feature_dir}.')

    # Dataset for training and validation
    train_set = GOYADataset(train_feature_path)
    val_set = GOYADataset(val_feature_path)

    dataset_sizes = {'train': len(train_set),  'val': len(val_set)}
            
    print('Training loader with %d samples' % dataset_sizes['train'])
    print('Validation loader with %d samples' % dataset_sizes['val'])
    
    model.to(device)
    # ------ ddp ---------
    model_without_ddp = model
    optimizer = torch.optim.Adam(model_without_ddp.parameters(), lr=args.lr)
    best_loss, best_content_loss, best_style_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx, model, optimizer = resume_epoch_model_ddp(args, model_without_ddp, optimizer)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False, broadcast_buffers=False)
        model_without_ddp = model.module
        sampler_train = DistributedSampler(train_set)
        sampler_val = DistributedSampler(val_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=sampler_val, num_workers=args.num_workers, shuffle=False)
    
    print("lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    loss_scaler = NativeScaler()

    # for soft content contrastive loss
    distance_matrix_path = 'data/generated/content_descr/content_descr_dist_mtx.pickle'
    with open(distance_matrix_path, 'rb') as handle:
        distance_matrix = pickle.load(handle)
    distance_matrix = torch.from_numpy(distance_matrix)
    print('Loaded distance matrix')

    main_loss = torch.nn.CosineEmbeddingLoss(margin=0.5)
    cls_loss = nn.CrossEntropyLoss()

    print('Start training...')
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.nepochs):
        # ----- ddp -------
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # -----------------
        best_loss, best_content_loss, best_style_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx = train_model_epoch(args, train_loader, val_loader, model, model_without_ddp, optimizer, epoch, device, best_loss, best_content_loss, best_style_loss, main_loss, cls_loss, best_batch_idx, best_content_batch_idx, best_style_batch_idx, loss_scaler, distance_matrix)

    end_time = time.time()
    print('Training took', str(round((end_time-start_time)/60, 2)), 'minutes')
    print('Finished Training')
