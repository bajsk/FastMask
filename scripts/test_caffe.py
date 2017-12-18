#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

caffe_root = os.path.dirname(os.path.realpath(__file__)) + "/../caffe-fm/"
sys.path.insert(0, caffe_root + '/build/install/python')

import caffe
