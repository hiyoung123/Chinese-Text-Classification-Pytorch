#!usr/bin/env python
#-*- coding:utf-8 -*-

import argparse
import os

from src.train import run_train
from src.evaluation import run_eval
from src.inference import run_inference
from utils.config import Config


def main(args):
    config = Config.parse(args.model)
    for k, v in vars(args).items():
        config.update({k: v})

    model_path = os.path.join(config.model_path, config.model + '_' + config.task_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    result_path = os.path.join(config.result_path, config.model + '_' + config.task_name)
    if not os.path.exists(model_path):
        os.mkdir(result_path)

    config.model_path = model_path + '/best_model.pt'
    config.result_path = result_path + '/result.csv'

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

    parser.add_argument('--init_weight', type=bool, default=False, help="Whether to use init_weight.")
    parser.add_argument('--loss_type', type=str, default='ce_loss', help="loss type, default=CE.")
    parser.add_argument('--lookahead', type=bool, default=False, help="Whether to use lookahead.")
    parser.add_argument('--fp16', type=bool, default=False, help="Whether to use fp16.")
    parser.add_argument('--fp16_opt_level', type=float, default=0.0, help="fp16_opt_level, default=0.0")
    parser.add_argument('--adv_type', type=str, default=None, help="adv_type, default=None")
    parser.add_argument('--flooding', type=float, default=0.0, help="flooding, default=0.0")
    parser.add_argument('--max_gradient_norm', type=float, default=0.0, help="max_gradient_norm, default=0.0")

    parser.add_argument('--pre_trained_model', type=str, default='pretrain/chinese_wwm_ext_pytorch', help="pre_trained")

    parser.add_argument('--save_by_step', type=bool, default=False, help="Whether save by step.")
    parser.add_argument('--log_step', type=int, default=10, help="Save model and log per step.")
    parser.add_argument('--log_dir', type=str, default='log/', help="log_dir, default=log/")
    parser.add_argument('--log_level', type=int, default=10, help='log_level, default=debug.')
    parser.add_argument('--patience', type=int, default=3, help="Patience num")
    parser.add_argument('--task_name', type=str, default='base', help="task name")
    parser.add_argument('--seed', type=int, default=7874, help='seed, default=7874')
    args = parser.parse_args()

    main(args)