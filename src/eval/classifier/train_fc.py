from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
import os
import pickle
import time
import numpy as np
import random
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import torch
from torch.utils.data import DataLoader, Dataset
from apex.parallel.LARC import LARC

from utils import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class PaintingFeatureDataset(Dataset):
    def __init__(self, data_pkl, class_type):
        '''
        Params: 
            data_pkl: pkl file of data feature
            class_type: content or style
        '''
        with open(data_pkl, 'rb') as handle:
            self.feature_all = pickle.load(handle)
            # img_idx is the correspondence idx of feature_all
            self.img_idx_list = pickle.load(handle)
            self.img_df = pickle.load(handle)

        self.feature_all = torch.tensor(self.feature_all.astype(np.float32)).view(self.feature_all.shape[0], -1)

        self.valid_df = self.img_df[self.img_df[class_type].notnull()]
        self.valid_label_list = self.valid_df[class_type].tolist()
        self.valid_index_list = self.img_df.index[self.img_df[class_type].notnull()].tolist()
        self.feature_valid = self.feature_all[self.valid_index_list]

    def __len__(self):
        return len(self.valid_label_list)

    def __getitem__(self, idx):
        feature = self.feature_valid[idx, :]
        label = self.valid_label_list[idx]

        return feature, torch.tensor(int(label))


def load_fc(args, model_resume_dir, model, optimizer):
    acc1 = 0
    args.start_epoch = 0
    if args.resume:
        if os.path.isdir(model_resume_dir):
            print("loading linear classifier")
            model_path = sorted(glob.glob(model_resume_dir + '/*'), key=os.path.getmtime)[-1]
            print("=> loading checkpoint '{}'".format(model_path))
            if args.gpu is None:
                checkpoint = torch.load(model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(model_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            acc1 = checkpoint['acc1']
            if args.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                acc1 = acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_resume_dir))
            acc1 = 0
    else:
        print("Not loading any checkpoint")

    return acc1, model, optimizer


def train_epoch(args, train_loader, model, optimizer, epoch, device, cls_loss):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()    
    end = time.time()
    
    for batch_idx, (features, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)
        
        features = features.to(device)
        targets = targets.to(device)

        outputs = model(features)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        loss = cls_loss(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), features.size(0))
        top1.update(acc1[0], features.size(0))
        top5.update(acc5[0], features.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            progress.display(batch_idx)

    if int(epoch) == 89:
        epoch_state_model = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc1': top1.avg,
            'optimizer': optimizer.state_dict(),
            }
        os.makedirs(args.model_dir, exist_ok=True)
        save_model_path = os.path.join(args.model_dir, '{}_{}_clf_epoch_{}.pth.tar'.format(args.model, args.space, str(epoch)))
        torch.save(epoch_state_model, save_model_path)
        print(f'model on epoch {epoch} saved to {save_model_path}')


def run_train(args):
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print('set seed', args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    train_data_pkl = os.path.join(args.feature_root, args.model, f'{args.model}_train_{args.space}_feature.pickle')
    print('training features from', train_data_pkl)

    train_set = PaintingFeatureDataset(train_data_pkl, args.space)
    train_loader = DataLoader(train_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True)

    dataset_sizes = {'train': len(train_set)}
    print('Training with %d samples' % dataset_sizes['train'])
    class_num = {'content': 10, 'style': 27} 

    model = torch.nn.Linear(args.in_feature, class_num[args.space])
    print(model)
    device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model.to(device) 
        
    cls_loss = torch.nn.CrossEntropyLoss()
    init_lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print("=> use LARS optimizer.")
    optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)
    acc1, model, optimizer = load_fc(args, args.model_dir, model, optimizer)

    for epoch in range(args.start_epoch, args.nepochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args.nepochs)
        train_epoch(args, train_loader, model, optimizer, epoch, device, cls_loss)
    
    print('Finished Training')

