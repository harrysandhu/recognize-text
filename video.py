#!/usr/bin/env python3
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2


# we'll be using videostream to access a webcam 
# and FPS to benchmark our frames per second for this script



def decode_prediction(scores, geometry):
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

    return (rects, confidences)



ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str,
    help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str, help="path to optional input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.6,
    help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
    help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
    help="resized image height (should be multiple of 32)")

args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
# load the pre-trained EAST text detector

net = cv2.dnn.readNet(args["east"])

vs = cv2.VideoCapture(args["video"])
fps = FPS().start()


while True:
    frame = vs.read()
    if frame is None:
        break
    frame = frame[1]
    # frame = imutils.resize(frame, width=1000)
    # orig = frame.copy()
    # fps.update()
    frame = imutils.rotate_bound(frame, 90)
    # cv2.imshow("Text Detection", rotated)

    frame = imutils.resize(frame, width=400)
    orig = frame.copy()

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)
    frame = cv2.resize(frame, (newW, newH))


    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                (100.68, 70.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (rects, confidences) = decode_prediction(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the frame
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    fps.update()
    # show the output frame
    cv2.imshow("Text Detection", orig)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.release()
cv2.destroyAllWindows()
