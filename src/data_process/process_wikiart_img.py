from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import pickle
import pandas as pd
import argparse
import pprint
import clip
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader, Dataset
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


parser = argparse.ArgumentParser(description='Exacting CLIP feature for WikiArt images')

parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--wikiart_dir', default='data/', type=str, help='directory to WikiArt')
parser.add_argument('--save_root', default='data/wikiart/img_feature', type=str, help='the directory to save the feature')


class PaintingDataset(Dataset):
    def __init__(self, img_df, wikiart_dir, transform=None):
        '''
        Params: 
            img_df: ['path', ...], absolute path
            transform: transform
        '''
        self.img_df = img_df
        self.img_path_list = self.img_df['path'].tolist()
        self.img_dir = wikiart_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_path_list[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, idx
 
    
def compute_painting_feature(args, img_df, data_type, batch_size=1024):
    '''compute feature of paintings'''
    # ---------- model ----------
    device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    model, preprocess = clip.load('ViT-B/32', device)
    os.makedirs(args.save_root, exist_ok=True)
    
    wikiart_dir = args.wikiart_dir
    
    # -------- dataset ----------
    dataset = PaintingDataset(img_df, wikiart_dir, transform=preprocess)
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            # pin_memory=True,
                            shuffle=False,
                            pin_memory=True
                            )

    out_all = torch.empty((0, 512), device=device)
    img_idx_all = torch.empty((0), device=device)
    
    with torch.no_grad():
        for x, img_idx in tqdm(data_loader):
            x = x.to(device)
            img_idx = img_idx.to(device)
            out_feature = model.encode_image(x)
            
            out_all = torch.vstack((out_all, out_feature))
            img_idx_all = torch.cat((img_idx_all, img_idx))
    
    save_path = os.path.join(args.save_root, f'wikiart_{data_type}_clip_feature.pickle')
    save_pickle(out_all, img_idx_all, img_df, save_path)
    print(f'{data_type} CLIP feature for WikiArt images is saved to {save_path}')

    return out_all, img_idx_all


def save_pickle(out, img_idx, img_df, pickle_path):
    with open(pickle_path, 'wb') as handle:
        pickle.dump(out.cpu().numpy(), handle)
        pickle.dump(img_idx.cpu().numpy(), handle)
        pickle.dump(img_df, handle)

    print('saved to pickle', pickle_path)


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as handle:
        out = pickle.load(handle)
        img_idx = pickle.load(handle)
        img_df = pickle.load(handle)

    return out, img_idx, img_df
    

if __name__ == '__main__':
    args = parser.parse_args()
    pprint.pprint(args.__dict__, indent=2)
    
    data_type_list = ['train', 'test']
    
    for data_type in data_type_list:
        image_csv = f'data/wikiart/info/wikiart_{data_type}.csv'
        print(f'{data_type} images from {image_csv}')
        img_df = pd.read_csv(image_csv)

        compute_painting_feature(args, img_df, data_type=data_type)
