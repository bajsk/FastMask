#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import rospy
from fast_mask_segmentation.srv import *
import cv2
import cv_bridge
import os

def fastmask_segmentation_client(img_msg):

    rospy.wait_for_service("/fast_mask_segmentation_server")
    
    try:
        fastmask_segmentation = rospy.ServiceProxy(
            "fast_mask_segmentation_server", FastMaskSegmentation)
        resp1 = fastmask_segmentation(img_msg)
        print ("Done with fm")
        return resp1

    except rospy.ServiceException, e:
        print ("Service call failed: %s" %e)

if __name__=="__main__":

    imgfile = os.path.dirname(os.path.realpath(__file__)) + "/../images/frame0000.jpg"
    img = cv2.imread(imgfile)
    br = cv_bridge.CvBridge()
    img_msg = br.cv2_to_imgmsg(img, encoding = "bgr8")
    
    bbox_arr = fastmask_segmentation_client(img_msg)
    print (bbox_arr.segmentation_bbox_arr.fm_bbox_arr[0])
