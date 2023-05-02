import os
import sys
import glob
import time
import json
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import cv2


imagesL = glob.glob("StereoL\*.jpg")
imagesR = glob.glob("StereoR\*.jpg")
#print(imagesL)
#print(imagesR)
CHECKERBOARD = (6,8)

def stereoCalibration(imagesL, imagesR, CHECKERBOARD):

    num_images = 0
    # Termination criteria for refining the detected corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    imgpointsL = []
    imgpointsR = []

    objpoints = []

    #Find Checkerboard points in each image
    for image in imagesL:
        i = imagesL.index(image)
        imgL = cv2.imread(imagesL[i])
        imgR = cv2.imread(imagesR[i])

        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        outputL = imgL.copy()
        outputR = imgR.copy()

        retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)
        retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)

        if retR and retL:
            num_images += 1
            objpoints.append(objp)

            cornersR = cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)

            cornersL = cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)

            cv2.drawChessboardCorners(outputR, CHECKERBOARD, cornersR, retR)
            cv2.drawChessboardCorners(outputL, CHECKERBOARD, cornersL, retL)

            cv2.imshow('cornersR', outputR)
            cv2.imshow('cornersL', outputL)
            cv2.waitKey(200)

        else:
            print("Checkerboard points not found in both images, remove: ", imagesL[i], imagesR[i])


    print("Number of good pairs: ", num_images)
    #cv2.destroyAllWindows()

    # Calibrating left camera
    print("Calibrating Left Camera")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, imgL_gray.shape[::-1], None, None)
    print("Left Camera Params: ", mtxL, distL, rvecsL, tvecsL)

    hL, wL = imgL_gray.shape[:2]
    #new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), -1, (wL, hL))


    # Calibrating right camera
    print("Calibrating Right Camera")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, imgR_gray.shape[::-1], None, None)
    print("Right Camera Params: ", mtxR, distR, rvecsR, tvecsR)

    hR, wR = imgR_gray.shape[:2]
   # new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),-1,(wR,hR))


    #Fix the intrinsic camara matrixes so that only Rot, Trns, E, and F are calculated.
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # Stereo Calibration
    print("Stereo Calibration...")
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR,
                                                                                        mtxL, distL, mtxR,
                                                                                        distR, imgL_gray.shape[::-1],
                                                                                        criteria_stereo, flags)

    #Stereo Rectification
    print("Stereo Rectification...")
    rectify_scale = 0
    rect_l, rect_r, proj_l, proj_r, Q, roiL, roiR = cv2.stereoRectify(mtxL, distL, mtxR, distR,
                                                                        imgL_gray.shape[::-1], Rot, Trns,
                                                                        rectify_scale, (0, 0))


    #Printing and Saving Calibration Data
    print('Saving stereo calibration data')
    stereo_calib_dict = {'mtxL': new_mtxL.tolist(), 'distL': distL.tolist(), 'mtxR': new_mtxR.tolist(), 'distR': distR.tolist(),
                         'Rot': Rot.tolist(), 'Trns': Trns.tolist(), 'Emat': Emat.tolist(), 'rect_l': rect_l.tolist(),
                         'rect_r': rect_r.tolist(), 'proj_r': proj_r.tolist(), 'proj_l': proj_l.tolist()}

    with open('stereocalibdata6.json', 'w') as fid:
        json.dump(stereo_calib_dict, fid, indent=2)



    print("Left Camera Matrix: \n", new_mtxL)
    print("Left Camera Distortion Params: \n", distL)
    print("Right Camera Matrix: \n", new_mtxR)
    print("Right Camera Distortion Params: \n", distR)
    print("Rotation Matrix: \n", Rot)
    print("Translation Vector: \n", Trns)
    print("Essential Matrix: \n", Emat)
    print("Fundamental Matrix: \n", Fmat)

    return mtxL, new_mtxL, distL, mtxR, new_mtxR, distR, Rot, Trns, Emat, Fmat

def stereoRectification(imgL, imgR, mtxL, distL, mtxR, distR, rect_l, rect_r, proj_l, proj_r):

    #Code to rectify and remap stereo images

    left_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    cv2.imshow('left', left_gray)
    cv2.imshow('right', right_gray)
    cv2.waitKey(0)

    #computing rectification map for left camera
    sizeL = left_gray.shape
    map1_L, map2_L =  cv2.initUndistortRectifyMap(mtxL, distL, rect_l, proj_l, sizeL, 11)

    #computing rectification map for right camera
    sizeR = right_gray.shape
    map1_R, map2_R = cv2.initUndistortRectifyMap(mtxR, distR, rect_r, proj_r, sizeR, 11)

    #remapping left and right images
    rectified_L = cv2.remap(left_gray, map1_L, map2_L, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    rectified_R = cv2.remap(right_gray, map1_R, map2_R, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

    cv2.imshow('rect L', rectified_L)
    cv2.imshow('rect R', rectified_R)
    cv2.waitKey(0)

    return rectified_L, rectified_R


#Uncomment below line when running StereoCalib, comment it out when running Depth Map
#stereoCalibration(imagesL, imagesR, CHECKERBOARD)