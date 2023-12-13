from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import glob
import os
import pickle
import time
import csv
import sys
import torch
from torch.utils.data import DataLoader, Dataset
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import gdown

from utils import AverageMeter, ProgressMeter, accuracy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class PaintingFeatureTestDataset(Dataset):
    def __init__(self, data_pkl, class_type):
        '''
        Params: 
            data_pkl: pickle file of data feature
            class_type: 'content', 'style'
        '''
        with open(data_pkl, 'rb') as handle:
            self.feature_all = pickle.load(handle)
            self.img_idx_list = pickle.load(handle)
            self.img_df = pickle.load(handle)
            
            self.label_list = self.img_df[class_type].tolist()
        
        self.feature_all = torch.tensor(self.feature_all.astype(np.float32)).view(self.feature_all.shape[0], -1)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        feature = self.feature_all[idx, :]
        label = torch.tensor(int(self.label_list[idx]))
        
        return feature, label


def read_csv(file_path):
    data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                model = row['model']
                data[model] = row
    
    return data


def write_to_csv(file_path, results, fieldnames):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(results.values())
        
        
def test(test_loader, model, device, cls_loss):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    
    model.eval()    
    
    with torch.no_grad():
        end = time.time()
        for batch_idx, (features, targets) in enumerate(test_loader):            
            features = features.to(device)       
            targets = targets.to(device)
            
            outputs = model(features)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            loss = cls_loss(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), features.size(0))
            top1.update(acc1[0], features.size(0))
            top5.update(acc5[0], features.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
                
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def run_test(args):
    test_data_pkl = os.path.join(args.feature_root, args.model, f'{args.model}_test_{args.space}_feature.pickle')
    print(f'Features are from {test_data_pkl}')
    
    test_set = PaintingFeatureTestDataset(test_data_pkl, args.space)
    test_loader = DataLoader(test_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=False)

    dataset_sizes = {'test': len(test_set)}
    print('Testing with %d samples' % dataset_sizes['test'])
    class_num = {'content': 10, 'style': 27} 

    model = torch.nn.Linear(args.in_feature, class_num[args.space])

    print('-----{}------'.format(args.space))
    print(model)

    device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model.to(device) 
        
    cls_loss = torch.nn.CrossEntropyLoss()
    
    if not os.path.isdir(args.model_dir):        
        print('Downloading linear classifier...')
        if args.model == 'GOYA':
            url = "https://drive.google.com/drive/folders/1rfDcnmswzyDWqe8cajL0TZvjZAWpdeA_"
            gdown.download_folder(url, output=args.model_root, quiet=True, use_cookies=False)
            print('Downloaded successfully!')
    
    print("Loading linear classifier")
    model_path = sorted(glob.glob(args.model_dir + '/*'), key=os.path.getmtime)[-1]
    print("=> Loading checkpoint '{}'".format(model_path))
    if args.gpu is None:
        checkpoint = torch.load(model_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(model_path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("=> Loaded checkpoint '{}' (epoch {})"
        .format(model_path, checkpoint['epoch']))

    acc1 = test(test_loader, model, device, cls_loss)
    
    result = {"model": args.model, args.space: acc1.cpu().numpy()}
    result_path = os.path.join(args.result_dir, 'clf_results.txt')
    existing_data = read_csv(result_path)

    model = result["model"]
    if model in existing_data:
        existing_data[model].update(result)
    else:
        existing_data[model] = result

    write_to_csv(result_path, existing_data, ["model", "content", "style"])

    print(f'Save result to {result_path}')
    
    return acc1
