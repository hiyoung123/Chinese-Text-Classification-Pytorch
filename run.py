#!usr/bin/env python
#-*- coding:utf-8 -*-

import argparse
import random

import numpy as np
import torch

from src.train import run_train
from src.evaluation import run_eval
from src.inference import run_inference
from utils.config import Config
from utils.log import Log


seed = 7874
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    config = Config.parse(args.model)
    config.update({
        'model': args.model,
        'log_step': args.log_step,
        'patience': args.patience,
        'save_by_step': args.save_by_step,
        'task_name': args.task_name
    })

    if args.do_train:
        run_train(config)

    if args.do_eval:
        run_eval(config)

    if args.do_predict:
        run_inference(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TextCNN', help="Model Type, default=TextCNN")
    parser.add_argument('--do_train', type=bool, default=True, help="Whether to run training.")
    parser.add_argument('--do_eval', type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument('--do_predict', type=bool, default=True, help="Whether to run predict on the test set.")
    parser.add_argument('--save_by_step', type=bool, default=False, help="Whether save by step")
    parser.add_argument('--log_step', type=int, default=10, help="Save model and log per step")
    parser.add_argument('--patience', type=int, default=3, help="Patience num")
    parser.add_argument('--task_name', type=str, default='base', help='task name')
    args = parser.parse_args()

    main(args)