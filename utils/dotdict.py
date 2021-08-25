#!/usr/bin/env python
# coding: utf-8

'''
this is dot style dictionary like JavaScript.
You can access via both bracket style and dot nonation style.

[example]

from dotdict import dotdict

d = dotdict({'a': 'spam', 'b': 'ham'})
print d['a']
>   spam
print d.b
>   ham
d.c = 'python!'
print d.c
>   python!
'''

__author__ = 'odiak'
__license__ = "Public Domain"
__version__ = "0.1"


class dotdict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__