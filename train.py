"""Launches routines that train (fewshot) recognition models on MiniImageNet.

Example of usage:
(1) For our proposed approach proposed:

    # 1st training stage: trains a cosine similarity based recognition model.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifier
    # 2nd training stage: finetunes the classifier of the recognition model and
    # at the same time trains the attention based few-shot classification weight
    # generator:
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1 # 1-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 # 5-shot case.

    All the configuration files that are used when launching the above
    training routines (i.e., miniImageNet_Conv128CosineClassifier.py,
    miniImageNet_Conv128CosineClassifierGenWeightAttN1.py, and
    miniImageNet_Conv128CosineClassifierGenWeightAttN5.py) are placed on the
    the directory ./config/

(2) For our implementations of the Matching Networks and Prototypical networks
    approaches:

    # Train the matching networks model for the 1-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN1

    # Train the matching networks model for the 5-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN5

    # Train the prototypical networks model for the 1-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN1

    # Train the prototypical networks model for the 5-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN5
"""

from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import MiniImageNet, FewShotDataloader

parser = argparse.ArgumentParser()   *模块：命令行选项、参数和子命令解析器；第一步，创建一个ArgumentParser对象
parser.add_argument('--config', type=str, required=True, default='',
    help='config file with parameters of the experiment. It is assumed that the'
         ' config file is placed on the directory ./config/.')   *第二步，添加参数；带有实验参数的配置文件
parser.add_argument('--checkpoint', type=int, default=0,
    help='checkpoint (epoch id) that will be loaded. If a negative value is '
         'given then the latest existing checkpoint is loaded.')*检查点
parser.add_argument('--num_workers', type=int, default=4,
    help='number of data loading workers')*数据加载数
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')*启用cuda
parser.add_argument('--disp_step', type=int, default=200,
    help='display step during training')*训练期间显示step
args_opt = parser.parse_args()*解析参数

exp_config_file = os.path.join('.','config', args_opt.config + '.py')*路径拼接，实验配置文件
exp_directory = os.path.join('.','experiments', args_opt.config)*实验目录

# Load the configuration params of the experiment加载实验的配置参数
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config*把源文件导入到config模块中
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
print("Loading experiment %s from file: %s" % (args_opt.config, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))*生成的日志、快照和模型文件将存储在

# Set train and test datasets and the corresponding data loaders设置训练和测试数据集以及相应的数据加载程序
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']

train_split, test_split = 'train', 'val'
dataset_train = MiniImageNet(phase=train_split)
dataset_test = MiniImageNet(phase=test_split)

dloader_train = FewShotDataloader(
    dataset=dataset_train,
    nKnovel=data_train_opt['nKnovel'],
    nKbase=data_train_opt['nKbase'],
    nExemplars=data_train_opt['nExemplars'], # num training examples per novel category
    nTestNovel=data_train_opt['nTestNovel'], # num test examples for all the novel categories
    nTestBase=data_train_opt['nTestBase'], # num test examples for all the base categories
    batch_size=data_train_opt['batch_size'],
    num_workers=args_opt.num_workers,
    epoch_size=data_train_opt['epoch_size'], # num of batches per epoch
)

dloader_test = FewShotDataloader(
    dataset=dataset_test,
    nKnovel=data_test_opt['nKnovel'],
    nKbase=data_test_opt['nKbase'],
    nExemplars=data_test_opt['nExemplars'], # num training examples per novel category
    nTestNovel=data_test_opt['nTestNovel'], # num test examples for all the novel categories
    nTestBase=data_test_opt['nTestBase'], # num test examples for all the base categories
    batch_size=data_test_opt['batch_size'],
    num_workers=0,
    epoch_size=data_test_opt['epoch_size'], # num of batches per epoch
)

config['disp_step'] = args_opt.disp_step
algorithm = alg.FewShot(config)
if args_opt.cuda: # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0: # load checkpoint
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else '*',
        train=True)

# train the algorithm
algorithm.solve(dloader_train, dloader_test)
