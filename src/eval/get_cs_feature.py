import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import pandas as pd
import argparse
import pprint
import gdown

import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset

from GOYA.src.model.net import make_GOYA, make_model_rm_clf, make_model_single_layer


parser = argparse.ArgumentParser(description='Exacting content and style features')
parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--model', default='GOYA', choices=['GOYA', 
                                        'model_ntxent', 'model_triplet',
                                        'model_contrastive_rm_clf', 'model_ntxent_rm_clf', 'model_triplet_rm_clf',
                                        'model_single_256', 'model_single_512', 'model_single_1024', 'model_single_2048',
                                        ], help='choose models')
parser.add_argument('--model_root', default='models/', type=str, help='directory for models')
parser.add_argument('--feature_root', default='evaluation/features/', type=str, help='directory for retrieving feature')


class PaintingFeatureDataset(Dataset):
    def __init__(self, data_pkl):
        '''
        Params: 
            data_pkl: pkl file of data feature
        '''
        with open(data_pkl, 'rb') as handle:
            self.feature_all = pickle.load(handle)
            # img_idx is the correspondence idx of feature_all
            self.img_idx_list = pickle.load(handle)
            _ = pickle.load(handle)

    def __len__(self):
        return len(self.img_idx_list)

    def __getitem__(self, idx):
        feature = self.feature_all[idx, :]
        return feature, idx


def extract_feature(img_df, data_pkl, data_type, space, batch_size=1024):
    '''compute feature of WikiArt paintings'''
    device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    if args.model in ['GOYA', 'model_ntxent', 'model_triplet']:
        model = make_GOYA()
        feature_dim = 2048
    elif args.model in ['model_contrastive_rm_clf', 'model_ntxent_rm_clf', 'model_triplet_rm_clf']:
        model = make_model_rm_clf()
        feature_dim = 2048
    elif args.model == 'model_single_256':
        model = make_model_single_layer(256)
        feature_dim = 256
    elif args.model == 'model_single_512':
        model = make_model_single_layer(512)
        feature_dim = 512
    elif args.model == 'model_single_1024':
        model = make_model_single_layer(1024)
        feature_dim = 1024
    elif args.model == 'model_single_2048':
        model = make_model_single_layer(2048)
        feature_dim = 2048

    dataset = PaintingFeatureDataset(data_pkl)
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            # pin_memory=True,
                            shuffle=False,
                            pin_memory=True
                            )
    
    model_path = os.path.join(args.model_root, args.model, f'{args.model}_{space}.pth.tar')
    if not os.path.exists(model_path):
        print('Downloading models...')
        if args.model == 'GOYA':
            url = "https://drive.google.com/drive/folders/1NDlW6JpmVFV6VDU8Cwsd28V_0v112i08"
            gdown.download_folder(url, output=os.path.join(args.model_root, args.model), quiet=True, use_cookies=False)
            print('Downloaded GOYA successfully!')
    
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from', model_path)

    for name, param in model.named_parameters():
        param.requires_grad = False    
    model.to(device) 
    model.eval()

    os.makedirs(os.path.join(args.feature_root, args.model), exist_ok=True)
    feature_path = os.path.join(args.feature_root, args.model, f'{args.model}_{data_type}_{space}_feature.pickle')
    
    out_content_all = torch.empty((0, feature_dim), device=device)
    out_style_all = torch.empty((0, feature_dim), device=device)
    # just make sure the index is correct if need check
    img_idx_all = torch.empty((0), device=device)
    
    with torch.no_grad():
        for x, img_idx in data_loader:
            x = x.to(device)
            img_idx = img_idx.to(device)
            
            if args.model in ['GOYA', 'model_ntxent', 'model_triplet']:
                _, _, feature_content_batch, feature_style_batch, _ = model(x)
            else:
                _, _, feature_content_batch, feature_style_batch = model(x)
            
            if space == 'content':
                out_content_all = torch.vstack((out_content_all, feature_content_batch))
            elif space == 'style':
                out_style_all = torch.vstack((out_style_all, feature_style_batch))

            img_idx_all = torch.cat((img_idx_all, img_idx))
    
    if space == 'content':
        save_pickle(out_content_all, img_idx_all, img_df, feature_path)
    elif space == 'style':
        save_pickle(out_style_all, img_idx_all, img_df, feature_path)

  
def save_pickle(out, img_idx, img_df, pickle_path):
    '''save them to a pickle file'''
    with open(pickle_path, 'wb') as handle:
        pickle.dump(out.cpu().numpy(), handle)
        pickle.dump(img_idx.cpu().numpy(), handle)
        pickle.dump(img_df, handle)

    print(f'Features are saved to {pickle_path}')


def load_pickle(pickle_path):
    '''load a pickle file'''
    with open(pickle_path, 'rb') as handle:
        out = pickle.load(handle)
        img_idx = pickle.load(handle)
        img_df = pickle.load(handle)

    return out, img_idx, img_df    


def check_wikiart_feature():
    wikiart_train_clip_feature_path = 'data/wikiart/img_feature/wikiart_train_clip_feature.pickle'
    wikiart_test_clip_feature_path = 'data/wikiart/img_feature/wikiart_test_clip_feature.pickle'

    if not os.path.exists(wikiart_train_clip_feature_path) or not os.path.exists(wikiart_test_clip_feature_path):
        print('Downloading features...')
        url = "https://drive.google.com/drive/folders/1LHhR0WvYRJTQT5Y-qMC4P6q0B8i7IzkG"
        gdown.download_folder(url, output='data/wikiart/img_feature', quiet=True, use_cookies=False)
        print('Downloaded!')


if __name__ == '__main__':
    args = parser.parse_args()
        
    pprint.pprint(args.__dict__, indent=2)
    
    if args.model == 'GOYA':
        check_wikiart_feature()
    
    data_type_list = ['train', 'test']
    space_list = ['content', 'style']

    for data_type in data_type_list:
        for space in space_list:
            print(f'-- Computing {data_type} feature in the {space} space --')
            image_csv = f'data/wikiart/info/wikiart_{data_type}.csv'
            print(f'{data_type} images from {image_csv}')
            # only for saving with feature
            img_df = pd.read_csv(image_csv)
            clip_feature_path = f'data/wikiart/img_feature/wikiart_{data_type}_clip_feature.pickle'
            print(f'{data_type} feature from {clip_feature_path}')
            extract_feature(img_df, clip_feature_path, data_type=data_type, space=space)
