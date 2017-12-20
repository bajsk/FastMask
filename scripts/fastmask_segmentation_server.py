#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import rospy
import cv_bridge
from fast_mask_segmentation.srv import *
from fast_mask_segmentation.msg import FastMaskBB2D, FastMaskBB2DArray

COLORS = [0xE6E2AF, 0xA7A37E, 0xDC3522, 0x046380, 
        0x468966, 0xB64926, 0x8E2800, 0xFFE11A,
        0xFF6138, 0x193441, 0xFF9800, 0x7D9100,
        0x1F8A70, 0x7D8A2E, 0x2E0927, 0xACCFCC,
        0x644D52, 0xA49A87, 0x04BFBF, 0xCDE855,
        0xF2836B, 0x88A825, 0xFF358B, 0x01B0F0,
        0xAEEE00, 0x334D5C, 0x45B29D, 0xEFC94C,
        0xE27A3F, 0xDF5A49]

class FastMaskSegmentationServer():
    
    def __init__(self, gpu_id = None):
        
        self.gpu_id = gpu_id
        self.model = "fm-res39"
        self.model_path = fast_mask_root + "/models/" + self.model + ".test.prototxt"
        self.weight_path = fast_mask_root + "/params/" + self.model + "_final.caffemodel"
        self.threshold = 0.85
        self.COLORS = COLORS
        self.br = cv_bridge.CvBridge()
        self.net = None
        self.save_and_display = rospy.get_param("~save_and_display", True)

        # load config
        if os.path.exists(fast_mask_root + "/configs/%szoom.json" % self.model):
            load_config(fast_mask_root + "/configs/%szoom.json" % self.model)
        else:
            print "Specified config does not exists, use the default config..."
        
        s = rospy.Service("fast_mask_segmentation_server", FastMaskSegmentation, self.handle_fast_mask_segmentation)
        rospy.spin()

    def handle_fast_mask_segmentation(self, req):

        if self.net == None:
            try:
                self.net = caffe.Net(self.model_path,
                                     self.weight_path,
                                     caffe.TEST)
            except:
                rospy.logerr("Error, cannot load fm net to the GPU")
                self.net = None
                self.service_queue = -1
                return FastMaskSegmentationResponse()

        try:
            image = self.br.imgmsg_to_cv2(req.rgb_img, desired_encoding = "bgr8")
            image = image.astype(np.float64)

            if self.gpu_id >= 0:
                caffe.set_mode_gpu()
                caffe.set_device(self.gpu_id)
            else:
                caffe.set_mode_cpu()

            oh, ow = image.shape[:2]
            im_scale = config.TEST_SCALE * 1.0 / max(oh, ow)
            input_blob = image - config.RGB_MEAN
            input_blob = input_blob.transpose((2, 0, 1))
            ih, iw = int(oh * im_scale), int(ow * im_scale)
            ih, iw = ih - ih % 4, iw - iw % 4
            input_blob = resize_blob(input_blob, dest_shape=(ih, iw))
            input_blob = input_blob[np.newaxis, ...]

            ret_masks, ret_scores = gen_masks(self.net, input_blob, config, dest_shape=(oh, ow))

            encoded_masks = encode(ret_masks)
            reserved = np.ones((len(ret_masks)))

            for i in range(len(reserved)):
                if ret_scores[i] < self.threshold:
                    reserved[i] = 0
                    continue
                if reserved[i]:
                    for j in range(i + 1, len(reserved)):
                        if reserved[j] and iou(encoded_masks[i], encoded_masks[j], [False]) > 0.5:
                            reserved[j] = 0

            temp_image = image.copy()
            fastmask_bbox_arr = FastMaskBB2DArray()

            for _ in range(len(ret_masks)):
                if ret_scores[_] > self.threshold and reserved[_]:

                    mask = ret_masks[_].copy()
                    bbox = toBbox(mask)
                    x, y, w, h = bbox

                    fastmask_bbox = FastMaskBB2D()

                    fastmask_bbox.bbox.x = x
                    fastmask_bbox.bbox.y = y
                    fastmask_bbox.bbox.w = w
                    fastmask_bbox.bbox.h = h

                    fastmask_bbox.score = ret_scores[_]
                    fastmask_bbox_arr.fm_bbox_arr.append(fastmask_bbox)

                    if self.save_and_display:
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
                        cv2.rectangle(temp_image, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (_color, _color, _color), 1)
            
            if self.save_and_display:
                image = image.astype(np.uint8)
                temp_image = temp_image.astype(np.uint8)
                cv2.imwrite(fast_mask_root + "/images/mask_result.jpg", image)
                cv2.imwrite(fast_mask_root + "/images/boundingbox_result.jpg", temp_image)
            
            self.net = None
            return FastMaskSegmentationResponse(segmentation_bbox_arr = fastmask_bbox_arr)

        except cv_bridge.CvBridgeError as e:
            rospy.logerr("CvBridge exception %s", e)
            return FastMaskSegmentationResponse()

if __name__ == '__main__':

    rospy.init_node("fast_mask_segmentation")
    FastMaskSegmentationServer(gpu_id = 0)
