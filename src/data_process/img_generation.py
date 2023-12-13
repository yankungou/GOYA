import torch
from diffusers import StableDiffusionPipeline
import pandas as pd
import os
import gdown
import argparse
import pprint
from tqdm import tqdm 


parser = argparse.ArgumentParser(description='Preprocessing content description for computing content contrastive loss during training.')

parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--save_root', default='data/generated/Image/', type=str)
parser.add_argument('--data_type', choices=['train', 'val'], type=str)


class NoWatermark:
    def apply_watermark(self, img):
        return img


def generate_img(args):
    model_id = "CompVis/stable-diffusion-v1-4"

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)

    pipe.watermark = NoWatermark()
    pipe = pipe.to(device)

    info_path = f'data/generated/info/path_prompt_{args.data_type}.csv'

    if not os.path.exists(info_path):
        print('Not found the file. Start downloading...')
        url = "https://drive.google.com/drive/folders/1ixD6oRRXsDXSPJr7Oc6dOXZMIUxhX2S0"
        gdown.download_folder(url, output='data/generated/info', quiet=True, use_cookies=False)
        print('Downloaded!')
        
    df_path_prompt = pd.read_csv(info_path, sep='\t')
    path_list = df_path_prompt.path.tolist()
    prompt_list = df_path_prompt.prompt.tolist()
    os.makedirs(args.save_root, exist_ok=True)

    print('Start to generate images...')
    for i, prompt in tqdm(enumerate(prompt_list)):
        image = pipe(prompt, num_inference_steps=50).images[0]  
        save_path = os.path.join(args.save_root, path_list[i])
        image.save(save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    pprint.pprint(args.__dict__, indent=2)
    
    generate_img(args)