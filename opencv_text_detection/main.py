import argparse
import os
import time

import cv2
from nms import nms
import numpy as np

import utils
from decode import decode
from draw import drawPolygons, drawBoxes
	
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def text_detection(image, east, min_confidence, width, height):
    image = cv2.imread(image)
    orig = image.copy()
    (origHeight, origWidth) = image.shape[:2]

    (newW, newH) = (width, height)
    ratioWidth = origWidth / float(newW)
    ratioHeight = origHeight / float(newH)

    image = cv2.resize(image, (newW, newH))
    (imageHeight, imageWidth) = image.shape[:2]

    layerNames = [ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3" ]

    # original slika
    cv2.imshow("Original", orig)

    print("[INFO] Učitavanje EAST text detektora...")
    net = cv2.dnn.readNet(east)

    blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    print("[INFO] Trajanje detekcije: {:.6f} sekundi".format(end - start))


    confidenceThreshold = min_confidence
    nmsThreshold = 0.4

    (rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

    offsets = []
    thetas = []
    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])


    functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]

    #print("[INFO] Running nms.boxes . . .")

    for i, function in enumerate(functions):

        start = time.time()
        indicies = nms.boxes(rects, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                                 nsm_threshold=nmsThreshold)
        end = time.time()

        indicies = np.array(indicies).reshape(-1)

        drawrects = np.array(rects)[indicies]

        name = function.__module__.split('.')[-1].title()
        print("[INFO] Trajanje izvođenja {} metode:  {:.6f} seconds i prondađeno je {} okvira".format(name, end - start, len(drawrects)))

        drawOn = orig.copy()
        drawBoxes(drawOn, drawrects, ratioWidth, ratioHeight, (0, 255, 0), 2)

        title = "nms.boxes {}".format(name)
        cv2.imshow(title,drawOn)
        cv2.moveWindow(title, 150+i*300, 350)

    cv2.waitKey(0)

    """
    polygons = utils.rects2polys(rects, thetas, offsets, ratioWidth, ratioHeight)

    print("[INFO] Running nms.polygons . . .")

    for i, function in enumerate(functions):

        start = time.time()
        indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                                 nsm_threshold=nmsThreshold)
        end = time.time()

        indicies = np.array(indicies).reshape(-1)

        drawpolys = np.array(polygons)[indicies]

        name = function.__module__.split('.')[-1].title()

        print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(name, end - start, len(drawpolys)))

        drawOn = orig.copy()
        drawPolygons(drawOn, drawpolys, ratioWidth, ratioHeight, (0, 255, 0), 2)

        title = "nms.polygons {}".format(name)
        cv2.imshow(title,drawOn)
        cv2.moveWindow(title, 150+i*300, 150)

    cv2.waitKey(0)
    """

def text_detection_command():
    imgs_path = "imgs"
    east = os.path.join(os.path.dirname(os.path.realpath("__file__")), 'frozen_east_text_detection.pb')
    min_confidence = 0.5
    width = 320
    height = 320

    for filename in os.listdir(imgs_path):
        text_detection(imgs_path + "/" + filename, east, min_confidence, width, height)
        break



if __name__ == '__main__':
    text_detection_command()
