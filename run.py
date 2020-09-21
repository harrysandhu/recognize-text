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
ap.add_argument("-c", "--min-confidence", type=float, default=0.6,
    help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
    help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
    help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# read the image
image = cv2.imread(args["image"])
#make a copy to output in the end
orig = image.copy()
# get image h and w
H, W = image.shape[:2]
# new h and w from the args 
newW, newH = (args["width"], args["height"])
#ratio old/new
rW = W / float(newW)
rH = H / float(newH)
# set new
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

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


net = cv2.dnn.readNet(args['east'])
"""
 construct a blob from the image and then perform a forward
 pass of the model to obtain th two output layer sets
 """
 # blob from image and preprocessing

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    (123.68, 116.78, 103.94), swapRB=True, crop=False)

start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)


# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores

(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

for y in range(0, numRows):

    # extract the scores, followed by the geometrical data 
    # used to derive the potential bounding box coords that surround text
    scoresData = scores[0,0, y]
    xData0 = geometry[0,0,y]
    xData1 = geometry[0,1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    for x in range(0, numCols):
        if scoresData[x] < args["min_confidence"]:
            continue
        
        #compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image

        (offsetX, offsetY) = (x * 4.0, y* 4.0)

        # extract the rotation angle  for the prediction and then 
        # compute the sin and cosine
        
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])


boxes = non_max_suppression(np.array(rects), probs=confidences)


for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    # draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (250, 250, 250, 0.8), 3)


end = time.time()


print("[INFO] text detection took {:.6f} seconds".format(end - start))

height, width = orig.shape[:2]

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', width, height)

cv2.imshow("img", orig)

cv2.waitKey(0)