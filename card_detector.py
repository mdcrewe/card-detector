############## Python-OpenCV Playing Card Detector ###############
#
# Description: Main file with algorithms to detect card and indicate rank and suit
#              from pi camera pictures. Image processing techniques are found in 
#              card_functions.py

import cv2
import numpy as np
import time
import os
import VideoStream as vs
import card_functions as cf

# -- Initialization of key parameters --

# Camera Parameters
IM_WIDTH = 1280
IM_HEIGHT = 720
RESOLUTION = (IM_WIDTH, IM_HEIGHT)
FRAME_RATE = 5

# Initializing video stream
videastream = vs.VideoStream(RESOLUTION, FRAME_RATE).start()
time.sleep(1)   # Need to give time for camera to start up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = cf.load_ranks( path + '/Card_Imgs/')
train_suits = cf.load_suits( path + '/Card_Imgs/')

quit = 0

while quit == 0:
  im = videostream.read()
  im_pre = cf.image_preprocess(im)
  counts_sort, count_card = cf.contours(im_pre)
  
  if len(counts_sort) != 0:
    cards = []
    j = 0
    
    for i in range(len(counts_sort)):
      if(count_card == 1):
        cards.append(cf.process_card(counts_sort[i], im))
        cards[j].best_rank_match, cards[j].best_suit_match,cards[j].rank_diff,cards[j].suit_diff = cf.match_card(cards[j],train_ranks,train_suits)
        
        # Draw center point and match result on the image.
        im = cf.draw_results(im, cards[j])
        j = j + 1
        
    if (len(cards) != 0):
      temp_cnts = []
      for i in range(len(cards)):
        temp_cnts.append(cards[i].contour)
      cv2.drawContours(im,temp_cnts, -1, (255,0,0), 2)
      
  cv2.imshow("Card Detector",im)
        
  # Poll the keyboard. If 'q' is pressed, exit the main loop.
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
      cam_quit = 1

cv2.destroyAllWindows()
videostream.stop()




