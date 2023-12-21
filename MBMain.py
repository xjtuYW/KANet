import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from MBTrainer import MBTrainer

from tools.utils import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# CEC 150 setp 40
def set_parameters(parser):
    parser.add_argument('--version', type=str, default='V1')
    parser.add_argument('--loss_base', type=float, default=0)
    parser.add_argument('--loss_inc', type=float, default=1.5)
    parser.add_argument('--loss_global', type=float, default=1)
    parser.add_argument('--full_data', type=str2bool, default=False)
    parser.add_argument('--enabel_grid_serach', type=str2bool, default=False)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('-gpu', default='0')

    # memory relevant
    parser.add_argument('--cap_layer', type=int, default=-1)
    parser.add_argument('--upd_layer', type=int, default=-1)
    parser.add_argument('--upd_targt', type=str, default='none')
    parser.add_argument('--enable_prompt', type=str2bool, default=False)
    parser.add_argument('--prompt_len', type=int, default=8)

    # general setting
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--seed', type=int, default=5) # 
    parser.add_argument('--storage_folder', type=str, default='tmpfile') # tmpfile
    parser.add_argument('--parent_folder', type=str, default='KANet') # tmpfile

    # optim setting
    parser.add_argument('--epoch', type=int, default=1,  # 50
                        help='base training epoches')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.1, # 0.1 
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--scheduler', type=str, default='MSLR')
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 40, 60, 80]) # [30, 40]
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=15) # 10

    # dataloader
    parser.add_argument('--state', type=str, default='train', choices=['train', 'test'],
                        help='training or testing')
    parser.add_argument('--network', type=str, default='ResNet18',
                        help='Encoding and Decoding images')
    parser.add_argument('--dataset', type=str, default='cub_200', help='datasets')
    parser.add_argument('--used_data', type=str, default=None,
                        help='the name of csv file which used to train or test the model')
    parser.add_argument('--sampler', type=str, default=None,
                        help='data sampler')
    parser.add_argument('--train_val_sNode', type=int, default=None,#4
                        help='how many data in each category is used as the trainig data')
    parser.add_argument('--workers', type=int, default=8,#4
                        help='num of thread to process image')
    # pretrain setting
    parser.add_argument('--batch_size', type=int, default=128) #128
    parser.add_argument('--temperature', type=float, default=16) # 16

    # second-stage training setting
    parser.add_argument('--train_flag', type=str2bool, default=False)
    parser.add_argument('--tasks', type=int, default=1)
    parser.add_argument('--n-way', type=int, default=10)
    parser.add_argument('--n-shot', type=int, default=5)
    parser.add_argument('--n-query', type=int, default=10)
    parser.add_argument('--topk', type=int, default=20)

    # test setting
    parser.add_argument('--batch_size_test', type=int, default=100) # 100

    return parser

def main_process(args):
    trainer = MBTrainer(args)
    trainer.train()
    trainer.test(reload=True, mode=args.mode, align_first_session=True, weight_update_mode='init')

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mynet')
    parser = set_parameters(parser)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    args.num_gpu = set_gpu(args)
    main_process(args)
