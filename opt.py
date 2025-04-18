import datetime
import argparse
import random
import numpy as np
import torch

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--num_iter', default=400, type=int, help='number of epochs for training')
        parser.add_argument('--fd', type=int, default=10, help='number of fold number')
        parser.add_argument('--hgc', type=int, default=64, help='hidden units of gconv layer')
        parser.add_argument('--lg', type=int, default=4, help='number of gconv layers')
        parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
        parser.add_argument('--edropout', type=float, default=0.3, help='edge dropout rate')
        parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--ckpt_path', type=str, default='./save_models/Model.pth', help='checkpoint path to save trained models')
        parser.add_argument('--saveLossAcc', type=str, default='./LossAcc.txt', help='save loss and acc to txt file')
        parser.add_argument('--saveFoldOut', type=str, default='./Log_EachFolds.txt', help='save fold outputs to txt file')
        parser.add_argument('--saveFinalOut', type=str, default='./Log_FinalAvg.txt', help='save final output to txt file')
        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
            print(" Using device:  CPU ")
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(" Using device:  GPU ")

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(123)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


