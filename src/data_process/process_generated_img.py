from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import pickle
import sys
import pandas as pd
import argparse
import pprint
import time
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import clip
import torch
from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser(description='Exacting CLIP features for diffusion generated image')

parser.add_argument('--gpu', default=0, type=int, help='which gpu')


class GeneratedImgDataset(Dataset):
    def __init__(self, img_df, transform=None):
        '''
        Params: 
            img_df: ['path', 'prompt'], relative path
            transform: transform
        '''
        self.img_df = img_df
        self.img_path_list = self.img_df['path'].tolist()
        self.transform = transform

    def get_abs_path(self, path):
        data_dir = 'data/generated/Image'
        path = data_dir + path
        return path
    
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img_path = self.get_abs_path(img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(idx)
    
    
def compute_image_feature(args, img_df, batch_size=2048):
    '''compute feature of generated images'''

    # -------- model ----------
    device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    model, preprocess = clip.load('ViT-B/32', device)

    pickle_save_dir = 'data/generated/img_feature'
    os.makedirs(pickle_save_dir, exist_ok=True)
    print('pickle will save to this directory:', pickle_save_dir)
    
    # -------- dataset ----------
    dataset = GeneratedImgDataset(img_df, transform=preprocess)

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
        for batch_idx, (x, img_idx) in enumerate(data_loader):
            x = x.to(device)
            img_idx = img_idx.to(device)
            out_feature = model.encode_image(x)

            out_all = torch.vstack((out_all, out_feature))
            img_idx_all = torch.cat((img_idx_all, img_idx))

            del x, out_feature

    pickle_path = os.path.join(pickle_save_dir, f'generated_{args.data_type}_clip_feature.pickle')
    save_pickle(out_all, img_idx_all, img_df, pickle_path)

    return out_all, img_idx_all


def save_pickle(out, img_idx, img_df, pickle_path):
    '''save them to a pickle file'''
    with open(pickle_path, 'wb') as handle:
        pickle.dump(out.cpu().numpy(), handle)
        pickle.dump(img_idx.cpu().numpy(), handle)
        pickle.dump(img_df, handle)

    print('saved to pickle', pickle_path)


def load_pickle(pickle_path):
    '''load a pickle file'''
    with open(pickle_path, 'rb') as handle:
        out = pickle.load(handle)
        img_idx = pickle.load(handle)
        img_df = pickle.load(handle)

    return out, img_idx, img_df
    

if __name__ == '__main__':
    args = parser.parse_args()

    pprint.pprint(args.__dict__, indent=2)

    data_type_list = ['val', 'train']
    
    for data_type in data_type_list:
        # these file can be found in https://drive.google.com/drive/folders/1ixD6oRRXsDXSPJr7Oc6dOXZMIUxhX2S0
        image_csv = f'data/generated/info/path_prompt_{data_type}.csv'
        img_df = pd.read_csv(image_csv, sep='\t')
        tic = time.time()
        compute_image_feature(args, img_df)
        toc = time.time()
        print(f"Computing feature of {len(img_df)} images spent {(toc - tic)/60:02f} minute(s).")
