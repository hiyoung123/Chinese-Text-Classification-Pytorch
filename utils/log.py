#!usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import logging
from logging.handlers import RotatingFileHandler
import colorlog

log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


def singleton(cls):
    # 单下划线的作用是这个变量只能在当前模块里访问,仅仅是一种提示作用
    # 创建一个字典用来保存类的实例对象
    _instance = {}

    def _singleton(*args, **kwargs):
        # 先判断这个类有没有对象
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)  # 创建一个对象,并保存到字典当中
        # 将实例对象返回
        return _instance[cls]

    return _singleton


@singleton
class Log:

    def __init__(self, log_dir='', log_level=logging.DEBUG):
        create_time = time.strftime("%Y%m%d%H%M%S")
        self.log_name = os.path.join(log_dir, create_time + '.log')
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

    def console(self, TAG, level, message):
        # 输出格式
        message = TAG + ': ' + message
        self.color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s][%(levelname)s] - %(message)s',
        log_colors=log_colors_config)  # 日志输出格式
        self.formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] - %(message)s')

        # 创建一个FileHandler，用于写到本地
        fh = logging.handlers.TimedRotatingFileHandler(self.log_name, when='MIDNIGHT', interval=1, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

        # 创建一个StreamHandler,用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.color_formatter)
        self.logger.addHandler(ch)

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        # 这两行代码是为了避免日志输出重复问题
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)
        fh.close()  # 关闭打开的文件

    def debug(self, TAG, message):
        self.console(TAG, 'debug', message)

    def info(self, TAG, message):
        self.console(TAG, 'info', message)

    def warning(self, TAG, message):
        self.console(TAG, 'warning', message)

    def error(self, TAG, message):
        self.console(TAG, 'error', message)

    def set_filename(self, file_name):
        self.log_name = file_name
