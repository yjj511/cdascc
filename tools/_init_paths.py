# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
# 把lib加入搜索路径，从而能够被eval()函数识别
lib_path = osp.join(this_dir, '..', 'lib')

add_path(lib_path)

# lib_path = osp.join(this_dir, '..', 'lib_cls')

add_path(lib_path)

work_path = '.'

add_path(work_path)