import pprint
import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def get_parser():
    parser = argparse.ArgumentParser(description='Classifier for content and style')
    
    parser.add_argument('--mode', type=str, help='Mode (train | test). If not set then first train then test.')
    parser.add_argument('--model', default='GOYA', choices=['GOYA', 
                                            'model_ntxent', 'model_triplet',
                                            'model_contrastive_rm_clf', 'model_ntxent_rm_clf', 'model_triplet_rm_clf',
                                            'model_single_256', 'model_single_512', 'model_single_1024', 'model_single_2048',
                                            ], help='choose models')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")    
    parser.add_argument('--space', choices=['content', 'style'], type=str, help='use content or style features')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', default=0.2, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--nepochs', default=90, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_root', default='evaluation/fc_models/', type=str, help='directory for saving model')
    parser.add_argument('--feature_root', default='evaluation/features/', type=str, help='directory for retrieving feature')
    parser.add_argument('--result_dir', default='evaluation/results/', type=str, help='path to save result')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float, metavar='W', help='weight decay (default: 0.)', dest='weight_decay')

    return parser


if __name__ == "__main__":
    
    # Load parameters
    parser = get_parser()    
    args, unknown = parser.parse_known_args()
    
    assert args.space in ['content', 'style'], 'Incorrect space type. Please select either content or style.'
    
    args.model_dir = os.path.join(args.model_root, args.model, args.space)
    os.makedirs(args.result_dir, exist_ok=True)

    if args.model == 'model_single_256':
        args.in_feature = 256
    elif args.model == 'model_single_512':
        args.in_feature = 512
    elif args.model == 'model_single_1024':
        args.in_feature = 1024
    else:
        args.in_feature = 2048
       
    pprint.pprint(args.__dict__, indent=2)
    
    # Run process
    if args.mode == None:
        from train_fc import run_train
        run_train(args)
        from test_fc import run_test
        run_test(args)
    elif args.mode == 'train':
        from train_fc import run_train
        run_train(args)
    elif args.mode == 'test':
        from test_fc import run_test
        run_test(args)

