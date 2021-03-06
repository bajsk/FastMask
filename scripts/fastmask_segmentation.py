import sys
import os
import argparse
import time
import cjson
from math import ceil

fast_mask_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
caffe_root = os.path.dirname(os.path.realpath(__file__)) + "/../caffe-fm/"
pycocotools_root = "/home/bjsk/publicWorkspace/dev/cocoapi/PythonAPI/"

sys.path.insert(0, caffe_root + "/build/install/python")
sys.path.insert(0, fast_mask_root + "/python_layers") 
sys.path.insert(0, fast_mask_root)
sys.path.insert(0, pycocotools_root)

import caffe
import config

import numpy as np
import cv2

from alchemy.utils.image import resize_blob, visualize_masks, load_image
from alchemy.utils.timer import Timer
from alchemy.utils.mask import encode, decode, crop, iou, toBbox
from alchemy.utils.load_config import load_config
from utils import gen_masks

'''
    python image_demo.py gpu_id model input_image
'''

COLORS = [0xE6E2AF, 0xA7A37E, 0xDC3522, 0x046380, 
        0x468966, 0xB64926, 0x8E2800, 0xFFE11A,
        0xFF6138, 0x193441, 0xFF9800, 0x7D9100,
        0x1F8A70, 0x7D8A2E, 0x2E0927, 0xACCFCC,
        0x644D52, 0xA49A87, 0x04BFBF, 0xCDE855,
        0xF2836B, 0x88A825, 0xFF358B, 0x01B0F0,
        0xAEEE00, 0x334D5C, 0x45B29D, 0xEFC94C,
        0xE27A3F, 0xDF5A49]

if __name__ == '__main__':
    
    # caffe setup
    caffe.set_mode_gpu()
    gpu_id = 0
    caffe.set_device(gpu_id)

    model = "fm-res39"
    init_weights = "fm-res39_final.caffemodel"
    input_image = fast_mask_root + "/images/frame0000.jpg"
    threshold = 0.85

    net = caffe.Net(
            fast_mask_root + 'models/' + model + '.test.prototxt',
            fast_mask_root + 'params/' + init_weights,
            caffe.TEST)

    # load config
    if os.path.exists("configs/%s.json" % model):
        load_config("configs/%szoom.json" % model)
    else:
        print "Specified config does not exists, use the default config..."

    image = cv2.imread(input_image)
    image = image.astype(np.float64)
    oh, ow = image.shape[:2]
    im_scale = config.TEST_SCALE * 1.0 / max(oh, ow)
    input_blob = image - config.RGB_MEAN
    input_blob = input_blob.transpose((2, 0, 1))
    ih, iw = int(oh * im_scale), int(ow * im_scale)
    ih, iw = ih - ih % 4, iw - iw % 4
    input_blob = resize_blob(input_blob, dest_shape=(ih, iw))
    input_blob = input_blob[np.newaxis, ...]

    ret_masks, ret_scores = gen_masks(net, input_blob, config, dest_shape=(oh, ow))

    # nms
    encoded_masks = encode(ret_masks)
    reserved = np.ones((len(ret_masks)))
    for i in range(len(reserved)):
        if ret_scores[i] < threshold:
            reserved[i] = 0
            continue
        if reserved[i]:
            for j in range(i + 1, len(reserved)):
                if reserved[j] and iou(encoded_masks[i], encoded_masks[j], [False]) > 0.5:
                    reserved[j] = 0

    temp_image = image.copy()

    
    for _ in range(len(ret_masks)):
        if ret_scores[_] > threshold and reserved[_]:
            mask = ret_masks[_].copy()
            bbox = toBbox(mask)
            bbox = [int(x) for x in bbox]

            mask[mask == 1] = 0.3
            mask[mask == 0] = 1
            color = COLORS[_ % len(COLORS)]
            _color = color & 0xff
            for k in range(3):
                image[:,:,k] = image[:,:,k] * mask
            mask[mask == 1] = 0
            mask[mask > 0] = 0.7
            for k in range(3):
                image[:,:,k] += mask * (color & 0xff)
                color >>= 8;
            cv2.rectangle(temp_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (_color, _color, _color), 1)
            
    image = image.astype(np.uint8)
    temp_image = temp_image.astype(np.uint8)
    cv2.imwrite("result.jpg", temp_image)
    cv2.imwrite("mask.jpg", mask)
