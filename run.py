#!usr/bin/env python
#-*- coding:utf-8 -*-

import argparse
from src.train import run_train
from src.evaluation import run_eval
from src.inference import run_inference
from utils.config import Config


def main(args):
    config = Config.parse(args.model)
    config.update({
        'model': args.model,
        'build_report': args.build_report,
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
    parser.add_argument('--build_report', type=bool, default=True, help="Whether to build report.")
    args = parser.parse_args()

    main(args)