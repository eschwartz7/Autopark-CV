# Autopark-CV

This is the github repository for the Stereo Camera aspect of the Autopark Project. 

Stereo Capture.py gives you basic code to capture images or video with the stereo camera and split the frames into left and right images. Change the image directories
to whatever you wish and use this script when collecting images for camera calibration.

StereoCalib.py contains the code needed to calibrate the stereo camera given two folders of images - one with right camera images, and one with left camera images.
stereocalibdata2.json contains the stereo camera intrinsic parameters should you ever need to reference them.

CentroidTracker3.py contains the code to run depth detection, YOLO object detection, and object tracking all within one script.
You will need to add Yolov3.cfg, stereocalibdata2.json, and Yolov3.weights to your directory in order the run the script.

