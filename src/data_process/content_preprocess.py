import clip
import torch
import pickle
import argparse
import pprint
import os
import pandas as pd
import gdown
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Preprocessing content description for computing content contrastive loss during training.')

parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--save_root', default='data/generated/content_descr/', type=str)


def compute_content_descr_feat(args):
    content_descr_path = 'data/generated/info/content_descr.csv'
    if not os.path.exists(content_descr_path):
        print('Not found the file. Start downloading...')
        url = "https://drive.google.com/drive/folders/1ixD6oRRXsDXSPJr7Oc6dOXZMIUxhX2S0"
        gdown.download_folder(url, output='data/generated', quiet=True, use_cookies=False)
        print('Downloaded!')

    content_descr_list = pd.read_csv(content_descr_path, header=None)[0].tolist()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device)

    text_features_all = torch.empty((0, 512), device=device)

    with torch.no_grad():
        for t in tqdm(content_descr_list):
            text = clip.tokenize([t]).to(device)
            text_features = model.encode_text(text)
            text_features_all = torch.vstack((text_features_all, text_features))

    os.makedirs(args.save_root, exist_ok=True)
    save_content_descr_feat_path = os.path.join(args.save_root, 'content_descr_clip_feature.pickle')
    with open(save_content_descr_feat_path, 'wb') as handle:
        pickle.dump(text_features_all.cpu().numpy(), handle)
    
    return text_features_all.cpu().numpy()
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    pprint.pprint(args.__dict__, indent=2)
    
    content_descr_features = compute_content_descr_feat(args)
    
    dist_matrix = cosine_distances(content_descr_features)
    
    save_dist_matrix_path = os.path.join(args.save_root, 'content_descr_dist_mtx.pickle')
    with open(save_dist_matrix_path, 'wb') as handle:
        pickle.dump(dist_matrix, handle)