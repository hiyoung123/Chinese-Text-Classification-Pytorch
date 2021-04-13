#!usr/bin/env python
#-*- coding:utf-8 -*-

try:
    import configparser as ConfigParser
except:
    import ConfigParser as ConfigParser

conf = ConfigParser.ConfigParser()
conf.read('config/base.cfg')


class Dict(dict):

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


class Config:
    @classmethod
    def parse(cls, section):
        config = {}
        for k, v in conf.items(section):
            config[k] = cls.parse_value(v)
        for k, v in conf.items('base'):
            config.update(
                {k: cls.parse_value(v)}
            )
        return Dict(config)

    @classmethod
    def parse_value(cls, v):
        flag = False

        try:
            v = int(v)
        except Exception as e:
            v = v
            flag = True

        if flag:
            try:
                v = float(v)
                flag = False
            except Exception as e:
                v = v
        if flag:
            if ',' in v:
                v = [int(x) for x in v.split(',')]
        return v


if __name__ == '__main__':
    print(Config.parse('TextCNN'))
