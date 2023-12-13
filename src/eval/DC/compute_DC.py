import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import argparse
import pickle
import csv


parser = argparse.ArgumentParser(description='Compute distance correlation (DC) between content and style')

parser.add_argument('--model', default='GOYA', choices=['GOYA', 
                                        'model_ntxent', 'model_triplet',
                                        'model_contrastive_rm_clf', 'model_ntxent_rm_clf', 'model_triplet_rm_clf',
                                        'model_single_256', 'model_single_512', 'model_single_1024', 'model_single_2048',
                                        ], help='choose models')
parser.add_argument('--feature_root', default='evaluation/features/', type=str, help='directory for retrieving feature')
parser.add_argument('--result_dir', default='evaluation/results/', type=str, help='path to save result')


def distance_correlation(A, B):
    #https://en.wikipedia.org/wiki/Distance_correlation
    #Input
    # A: the first variable
    # B: the second variable
    # The numbers of samples in the two variables must be same.
    #Output
    # dcor: the distance correlation of the two samples

    n = A.shape[0]
    if B.shape[0] != A.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(A))
    b = squareform(pdist(B))
    T1 = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    T2 = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    #Use Equation 2 to calculate distance covariances.
    dcov_T1_T2 = (T1 * T2).sum() / float(n * n)
    dcov_T1_T1 = (T1 * T1).sum() / float(n * n)
    dcov_T2_T2 = (T2 * T2).sum() / float(n * n)

    #Equation 1 in the paper.
    dcor = np.sqrt(dcov_T1_T2) / np.sqrt(np.sqrt(dcov_T1_T1) * np.sqrt(dcov_T2_T2))
    return dcor


def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as handle:
        feat = pickle.load(handle)

    return feat


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
        

if __name__ == "__main__":
    
    # Load parameters
    args, unknown = parser.parse_known_args()
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Load the data directory and saving path
    content_test_path = os.path.join(args.feature_root, args.model, f'{args.model}_test_content_feature.pickle')
    style_test_path = os.path.join(args.feature_root, args.model, f'{args.model}_test_style_feature.pickle')
        
    print('Content feature from', content_test_path)
    print('Style feature from', style_test_path)

    content_feat = read_pkl(content_test_path)
    style_feat = read_pkl(style_test_path)

    print('Start distance correlation test on content and style...')
    dis_correlation = distance_correlation(content_feat, style_feat)
    print(f'Distance correlation of content and style in model {args.model} is {dis_correlation}.')
    
    result = {'model': args.model, 'DC': dis_correlation}
    save_path = os.path.join(args.result_dir, 'dc_results.txt')
    existing_data = read_csv(save_path)

    model = result["model"]
    if model in existing_data:
        existing_data[model].update(result)
    else:
        existing_data[model] = result

    write_to_csv(save_path, existing_data, ["model", "DC"])
    print(f'Results are saved to {save_path}!')