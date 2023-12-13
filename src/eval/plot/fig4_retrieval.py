import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import os
import numpy as np
import pandas as pd
import argparse
import pprint
from sklearn.metrics.pairwise import cosine_distances


parser = argparse.ArgumentParser(description='Similarity retrieval')

parser.add_argument('--query', default=0, type=int, choices=[0,1,2,3], help='the query index in the figure 4 in the paper')
parser.add_argument('--wikiart_dir', default='data/', type=str, help='directory to WikiArt')
parser.add_argument('--save_dir', default='evaluation/figure', type=str, help='directory to save the figures')


def get_dist_mtx(data_pkl):
    with open(data_pkl, 'rb') as handle:
        feature_all = pickle.load(handle)
        _ = pickle.load(handle)
        _ = pickle.load(handle)

    distances = cosine_distances(feature_all)

    return distances
    
    
def get_closest_path_list(args, distances, index, img_df, k=5):
    closest_idx_list = np.argsort(distances[index, :])[1: k+1]
    vis_path_list = [img_df['path'][index]] + img_df['path'][closest_idx_list].tolist()
    vis_path_list = [os.path.join(args.wikiart_dir, path) for path in vis_path_list]
    
    return vis_path_list


def vis_closest(args, vis_input_list):

    fig, axes = plt.subplots(nrows=2, ncols=11, figsize=(110, 10))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0.3, right=1)
    [ax.axis("off") for ax in np.ravel(axes)]
    fontsize = 40

    for row, (axis, data) in enumerate(zip(axes, vis_input_list)):
        (ax_a, ax_sim_0, ax_sim_1, ax_sim_2, ax_sim_3, ax_sim_4, ax_sim_5, ax_sim_6, ax_sim_7, ax_sim_8, ax_sim_9) = axis
        
        sim_0 = Image.open(data[1])
        ax_sim_0.imshow(sim_0)
        
        sim_1 = Image.open(data[2])
        ax_sim_1.imshow(sim_1)
        
        sim_2 = Image.open(data[3])
        ax_sim_2.imshow(sim_2)

        sim_3 = Image.open(data[4])
        ax_sim_3.imshow(sim_3)
        
        sim_4 = Image.open(data[5])
        ax_sim_4.imshow(sim_4)

        sim_5 = Image.open(data[6])
        ax_sim_5.imshow(sim_5)
        
        sim_6 = Image.open(data[7])
        ax_sim_6.imshow(sim_6)
        
        sim_7 = Image.open(data[8])
        ax_sim_7.imshow(sim_7)
        
        sim_8 = Image.open(data[9])
        ax_sim_8.imshow(sim_8)
        
        sim_9 = Image.open(data[10])
        ax_sim_9.imshow(sim_9)
        
        if int(row) % 2 == 0:
            space = 'content'
            color = 'tomato'
            a_image = Image.open(data[0])
            ax_a.imshow(a_image)
            ax_a.set_title(f"Query {args.query}", fontsize=fontsize)
        elif int(row) % 2 == 1:
            space = 'style'
            color = 'cornflowerblue'

        ax_sim_0.set_title(f'{space}, 1st', fontsize=fontsize, color=color)
        ax_sim_1.set_title(f'{space}, 2nd', fontsize=fontsize, color=color)
        ax_sim_2.set_title(f'{space}, 3rd', fontsize=fontsize, color=color)
        ax_sim_3.set_title(f'{space}, 4th', fontsize=fontsize, color=color)
        ax_sim_4.set_title(f'{space}, 5th', fontsize=fontsize, color=color)
        ax_sim_5.set_title(f'{space}, 6th', fontsize=fontsize, color=color)
        ax_sim_6.set_title(f'{space}, 7th', fontsize=fontsize, color=color)
        ax_sim_7.set_title(f'{space}, 8th', fontsize=fontsize, color=color)
        ax_sim_8.set_title(f'{space}, 9th', fontsize=fontsize, color=color)
        ax_sim_9.set_title(f'{space}, 10th', fontsize=fontsize, color=color)

    fig.tight_layout()
    
    fig_save_path = os.path.join(args.save_dir, f'fig4_query_{args.query}.png')
    plt.savefig(fig_save_path, dpi=100)
    print(f'The similarity retrieval result of the query {args.query} in Figure 4 was saved to {fig_save_path}')
    
    return fig


if __name__ == '__main__':
    args = parser.parse_args()
    pprint.pprint(args.__dict__, indent=2)

    content_feature_path = 'evaluation/features/GOYA/GOYA_test_content_feature.pickle'
    style_feature_path = 'evaluation/features/GOYA/GOYA_test_style_feature.pickle'
    
    content_dist_mtx = get_dist_mtx(content_feature_path)
    style_dist_mtx = get_dist_mtx(style_feature_path)
    
    image_test_csv = 'data/wikiart/info/wikiart_test.csv'
    print(f'test images from {image_test_csv}')
    img_df = pd.read_csv(image_test_csv)
    
    index_list = [5638, 9718, 9837, 12000]
    index = index_list[args.query]
    content_closet_path_list = get_closest_path_list(args, content_dist_mtx, index, img_df, 10)
    style_closet_path_list = get_closest_path_list(args, style_dist_mtx, index, img_df, 10)

    vis_input_list = [content_closet_path_list, style_closet_path_list]
    info_df = pd.DataFrame(vis_input_list)
    os.makedirs(args.save_dir, exist_ok=True)
    
    vis_closest(args, vis_input_list)