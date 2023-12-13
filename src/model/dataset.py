import os
import pandas as pd
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset


def read_data(data):
    '''input: dataframe or csv path
       output: dataframe
    '''
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, str) and os.path.isfile(data):
        return pd.read_csv(data, on_bad_lines='skip')


class GOYADataset(Dataset):
    def __init__(self, feature_pkl):
        '''
        Params: 
            feature_pkl: feature_all, index, dataframe of path and prompt
        '''
        with open(feature_pkl, 'rb') as handle:
            self.feature_all = pickle.load(handle)
            # img_idx is the correspondence idx of feature_all
            self.img_idx_list = pickle.load(handle)
            self.path_prompt_df = pickle.load(handle)
        
        self.path_prompt_df[['content', 'style']] = self.path_prompt_df['prompt'].str.split(', ', expand=True)
        self.style_list = self.path_prompt_df['style'].tolist()
        self.style_dict = self.get_style_dict()
        self.content_list = self.path_prompt_df['content'].tolist()
        self.content_dict = self.get_content_dict()

    def get_style_dict(self): 
        style_path = 'data/generated/info/style.txt'
        style_df = pd.read_csv(style_path, header=None, sep=' ', names=['number', 'style'])
        style_list = style_df['style'].tolist()
        style_list = [s.replace('_', ' ') for s in style_list]
        style_df['style'] =style_list
        style_dict = dict(style_df.values)
        style_dict = {v: k for k, v in style_dict.items()}
        
        return style_dict

    def get_content_dict(self):
        title_csv_file = 'data/generated/info/content_descr.csv'
        title_list = pd.read_csv(title_csv_file, header=None)[0].tolist()
        title_num_list = [*range(len(title_list))]
        title_dict = dict(zip(title_list, title_num_list))
        return title_dict
        
    def __len__(self):
        return len(self.style_list)

    def __getitem__(self, idx):
        feature = self.feature_all[idx, :]
        feature = torch.from_numpy(feature)
        style = self.style_dict[self.style_list[idx]]
        content = self.content_dict[self.content_list[idx]]
        
        return feature, (torch.tensor(int(content)), torch.tensor(int(style)))
