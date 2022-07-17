############## Python-OpenCV Playing Card Detector ###############
#
# Author: Matthew Crewe
# Date: 9/5/17
# Description: Main file with all image processing techniques to detect card and indicate rank and suit
#              from pi camera pictures

import cv2
import numpy as np
import time
import os
from picamera import PiCamera
from picamera.array import PiRGBArray

# -- Initialization of key parameters --

# Camera Parameters
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 5

# Setting up camera object



