import os
import sys
import glob
import time
import json
import numpy as np
from scipy import io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cv2
import time
from scipy.spatial import distance as dist
from collections import OrderedDict
import serial

#Inspiration for centroid tracking taken from PyImageSearch: https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/



class CentroidTracker():
    def __init__(self, maxDisappeared=15):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given object is allowed to be marked as "disappeared"
        self.maxDisappeared = maxDisappeared

    def register(self, world_pt):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = world_pt
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # deregister an object ID
        if objectID in self.objects.keys():
            del self.objects[objectID]
        if objectID in self.disappeared.keys():
            del self.disappeared[objectID]

    # rects = (startX, startY, endX, endY)
    def update(self, world_pts):

        # check to see if the list of input bounding box rectangles is empty
        if len(world_pts) == 0:

            #mark existing objects as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # deregister object if it is missing for > n frames
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(world_pts), 4), dtype="float")
        # loop over the bounding box rectangles
        for (i, (x, y, z, l)) in enumerate(world_pts):
            # use the bounding box coordinates to derive the centroid
            inputCentroids[i] = (x, y, z, l)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        #try to match new centroids to existing centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                #Get previous coordinates and set new object centroid
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0


                usedRows.add(row)
                usedCols.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                #If there are less input centroids than previously, check if objects have disappeared
                if D.shape[0] >= D.shape[1]:
                    # loop over the unused row indexes
                    for row in unusedRows:
                        # grab the object ID for the corresponding row index
                        objectID = objectIDs[row]
                        if objectID in self.disappeared.keys():
                            self.disappeared[objectID] += 1
                            # check to see if object should be deregistered
                            if self.disappeared[objectID] > self.maxDisappeared:
                                self.deregister(objectID)

                #Otherwise register new centroids
                else:
                    for col in unusedCols:
                        self.register(inputCentroids[col])
        # return the set of trackable objects
        return self.objects


def stereoRectification(imgL, imgR, map1_L, map2_L, map1_R, map2_R):
    # Code to rectify and remap stereo images

    left = imgL
    right = imgR

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # remapping left and right images INTER_LANCZOS4 VS LINEAR
    rectified_L = cv2.remap(left, map1_L, map2_L, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    rectified_R = cv2.remap(right, map1_R, map2_R, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=21)
    disparity = stereo.compute(rectified_L, rectified_R)

    #Create WLS filter
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    disparity_right = right_matcher.compute(rectified_R, rectified_L)

    filt = cv2.ximgproc.createDisparityWLSFilter(stereo)
    filt.setLambda(8000)
    filt.setSigmaColor(1.5)

    filt_disp = filt.filter(disparity, rectified_L, disparity_map_right = disparity_right)

    return filt_disp, rectified_L

def obj_tracker():

    # Load Stereo Camera Parameters
    f = open('stereocalibdata2.json', )
    stereo_data = json.load(f)

    [mtxL, distL, mtxR, distR, Rot, Trns, Emat, Fmat, rect_l, rect_r, proj_l, proj_r] = stereo_data.values()

    mtxL = np.asarray(mtxL, dtype=np.float64)
    distL = np.asarray(distL, dtype=np.float64)
    mtxR = np.asarray(mtxR, dtype=np.float64)
    distR = np.asarray(distR, dtype=np.float64)
    Rot = np.asarray(Rot, dtype=np.float64)
    Trns = np.asarray(Trns, dtype=np.float64)
    Emat = np.asarray(Emat, dtype=np.float64)
    Fmat = np.asarray(Fmat, dtype=np.float64)
    rect_l = np.asarray(rect_l, dtype=np.float64)
    rect_r = np.asarray(rect_r, dtype=np.float64)
    proj_l = np.asarray(proj_l, dtype=np.float64)
    proj_r = np.asarray(proj_r, dtype=np.float64)


    # Get camera attributes
    cam_width = 1280
    cam_height = 480
    nsize = np.asarray((cam_width / 2, cam_height), dtype=np.uint32)

    # Get stereo mappings
    rect_l, rect_r, proj_l, proj_r, Q, roiL, roiR = cv2.stereoRectify(mtxL, distL, mtxR, distR, nsize, Rot, Trns,
                                                                      flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    map1_L, map2_L = cv2.initUndistortRectifyMap(mtxL, distL, rect_l, proj_l, nsize, 11)
    map1_R, map2_R = cv2.initUndistortRectifyMap(mtxR, distR, rect_r, proj_r, nsize, 11)

    #Initialize Centroid Tracker object
    ct = CentroidTracker()

    """
    #Set up serial connection
    ser = serial.Serial('COM5', 9600, timeout=0.1)
    ser.close()
    ser.open()
    """

    # Initialize YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    print(net.getUnconnectedOutLayers())
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #Initialize video object
    vid = cv2.VideoCapture(1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #Record captured frames if you want
    #video = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MP42'),2, (640, 480))
    pts_dict = OrderedDict()
    total_time = 0
    frame_count = 0
    while (True):
        start_time = time.time()
        ret, frame = vid.read()

        frame = cv2.resize(frame, (1280, 480))

        imgL = frame[0:480, 0:int(1280 / 2)]
        imgR = frame[0:480, int(1280 / 2):1280]

        disparityMap, rectified_L = stereoRectification(imgL, imgR, map1_L, map2_L, map1_R, map2_R)

        # Loading image
        img = cv2.cvtColor(rectified_L, cv2.COLOR_GRAY2RGB)
        # img = cv2.resize(img, None, fx=1, fy=1)

        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        centers1 = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    centers1.append((center_x, center_y))
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        centers = []
        world_pts = []
        labels = []
        new_boxes = []
        font = cv2.FONT_HERSHEY_PLAIN
        #Only use boxes filtered through NMS
        for i in range(len(boxes)):
            if i in indexes:
                centers.append(centers1[i])
                x, y, w, h = boxes[i]
                new_boxes.append(boxes[i])
                disparity_box = disparityMap[y:y + h, x:x + w]
                disp_values = disparity_box.flatten()
                disp_values = np.delete(disp_values, np.where(disp_values < 2))
                labels.append(classes[class_ids[i]])
                color = colors[class_ids[i]]
                #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                if disp_values.size > 0:
                    avg = np.average(disp_values)
                    weights_lst = []

                    # Weight values such that larger disparities (closer) are given higher weight
                    for disparity in disp_values:
                        weights_lst.append(1 + (disparity - avg)/avg*2)
                    disparity = np.average(disp_values, weights = weights_lst)

                    obj_depth = ((3.28 * 420 * 70.1) / ((disparity / 16) * 1000))

                    #get real-world x and y coordinates of bounding box centroid
                    world_pt = np.asarray([[centers1[i][0]], [centers1[i][1]], [1]])
                    world_pt = obj_depth * np.matmul(np.linalg.inv(mtxL), world_pt)

                    #Include class ID in world point to filter out bounding box crossings
                    world_pts.append((world_pt[0][0], world_pt[1][0], world_pt[2][0], class_ids[i] * 10))

                else:
                    world_pts.append((0, 0, 0, 0))
                    labels.append(classes[class_ids[i]])


        # update our centroid tracker using the computed set of bounding boxes
        objects = ct.update(world_pts)

        throttle_val = 1
        # loop over the tracked objects
        for (objectID, world_pt) in objects.items():
            # draw both the ID of the object and the centroid of the object
            text = "ID {}".format(objectID)
            #print("object coordinates: " + str(world_pt))

            #If object matches with one currently detected, get its label and display info
            for i in range(0, len(world_pts)):
                #print("detected coordinates: " + str(world_pts[i]))
                comparison = np.asarray(world_pts[i]) == np.asarray(world_pt)
                if comparison.all() and world_pts[i][2] != 0:
                    #print("object detected")
                    label = labels[i] + " " + "%.1f" % world_pts[i][2] + " ft"
                    x, y, w, h = new_boxes[i]
                    #cv2.putText(img, text, (centers[i][0] - 10, centers[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                #(0, 255, 0), 1)
                    #Display label in red if object is in ROI (stop)
                    if -3.5 < world_pts[i][0] < 3 and -4.5 < world_pts[i][1] < 6 and world_pts[i][2] < 15:
                        if labels[i] == "person":
                            cv2.putText(img, label, (x, y - 15), font, 1, (0, 0, 255), 1)
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        throttle_val = 0
                    #Display label in yellow if object may enter ROI (slow)
                    elif objectID in pts_dict.keys():
                        x_diff = world_pt[0] - pts_dict[objectID][0]
                        predicted_x = (world_pt[0] + x_diff)

                        if -3.5 < predicted_x < 3:
                            if labels[i] == "person":
                                cv2.putText(img, label, (x, y - 15), font, 1, (0, 255, 255), 1)
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            throttle_val = min(0.5, throttle_val)
                            send_throttle = True

                        else:
                            if labels[i] == "person":
                                cv2.putText(img, label, (x, y - 15), font, 1, (0, 255, 0), 1)
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #Otherwise display label in green (continue)
                    else:
                        if labels[i] == "person":
                            cv2.putText(img, label, (x, y - 15), font, 1, (0, 255, 0), 1)
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    pts_dict[objectID] = world_pt

        #Send throttle value to Jetson Nano through serial connection
        #print("Sent throttle val: ", throttle_val)
        #ser.write(str(throttle_val).encode())

        #video.write(img)
        cv2.namedWindow('Yolo Tracking Applied', cv2.WINDOW_NORMAL)
        cv2.imshow("Yolo Tracking Applied", img)

        end_time = time.time()
        total_time += end_time - start_time
        frame_count += 1
        print("time elapsed: ", end_time - start_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    print("Avg time per frame: ", total_time / frame_count)

obj_tracker()