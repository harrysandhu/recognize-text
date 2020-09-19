#!/usr/bin/env python3
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, 
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=64,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=64,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])

# get image h and w
H, W = image.shape[:2]
# new h and w from the args
newW, newH = (args["width"], args["height"])
#ratio old/new
rW = W / float(newW)
rH = H / float(newH)
# set new
image = cv2.resize(image, (newW, newH))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)
cv2.imshow('image', image)
cv2.waitKey()


# define the two output layer names for the EAST detector model
# first one output probs
# second one - bounding box coords of text

"""
The first layer is our output sigmoid activation 
which gives us the probability of a region containing text or not.

The second layer is the output feature map that represents the 
“geometry” of the image — we’ll be able to use this geometry
 to derive the bounding box coordinates of the text in the input image
"""
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]


