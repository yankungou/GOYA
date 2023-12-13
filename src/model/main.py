import os
import argparse
import pprint


def get_parser():
    parser = argparse.ArgumentParser(description='Train GOYA')

    parser.add_argument('--model', default='GOYA', choices=['GOYA', 
                                            'model_ntxent', 'model_triplet',
                                            'model_contrastive_rm_clf', 'model_ntxent_rm_clf', 'model_triplet_rm_clf',
                                            'model_single_256', 'model_single_512', 'model_single_1024', 'model_single_2048',
                                            ], help='choose models')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gpu', default='0', type=int, help='Which gpu if not using DDP')

    parser.add_argument('--model_root', default='models/', type=str, help='directory for saving trained models')
    parser.add_argument('--feature_dir', default='data/generated/img_feature/', type=str, help='directory for loading train and val features')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate decay.")

    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--nepochs', default=100, type=int)
    parser.add_argument('--margin', default=0.5, type=float)
    
    # DDP
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--resume', default='', type=str, help='The path of which point to start. It could be a directory or file.')

    return parser


if __name__ == "__main__":
    
    # Load parameters
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    pprint.pprint(args.__dict__, indent=2)

    os.makedirs(args.model_root, exist_ok=True)

    if args.model == 'GOYA':
        from train_GOYA import train
        train(args)
    else:
        from train_others import train
        train(args)
