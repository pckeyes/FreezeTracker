# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:40:51 2017

@author: pckeyes
"""
#DESCRIPTION:
#This script can be used to detect lack of motion in a video file using
#frame subtraction. The current frame is compared to a previous frame by
#finding the difference between each pixel in the two frames.
#
#Parameters:
#n_prev_frames:         Decides how many frames will be between the current 
#                       frame and the frame to which the current frame will 
#                       be compared
#black_threshold:       Decides what pixel value will be considered "black";
#                       in other words, sets the threshold for what is 
#                       considered an unchanged pixel
#n_pixels_threshold:    Decides how many pixels can be above "black_threshold"

import numpy as np
import cv2

#read in first n frames of video
cap = cv2.VideoCapture("/Users/piperkeyes/Desktop/fc_test_tone.mp4")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
prev_frames = []
n_prev_frames = 4
for i in range(0,n_prev_frames):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    prev_frames.append(frame)
    
#lists to store freezing data
diff_indices = []
diff_frames = []

#threshold params
black_threshold = 20
n_pixels_threshold = 100

#text params
placement = (20,20)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
color = (255,255,255)
line_type = 2
count = 0

#read in the video frame by frame and decide whether mouse is not moving ("frozen")
for frame_n in range(n_frames - n_prev_frames):
    print(count)
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    diff = cv2.absdiff(frame,prev_frames[0])
    cv2.imshow("frame", frame)
    cv2.imshow("subtraction", diff)
    
    #update which frames will be used as "prev_frame"
    prev_frames = prev_frames[1:]
    prev_frames.append(frame)
    
    #flatten the image to a 2D vector
    flattened_diff = diff.sum(2)
    diff_frames.append(flattened_diff)
    indices, _ = np.where(flattened_diff > black_threshold)    
    n_diff_pixels = len(indices)
    
    #if the number of "non-black" pixels is less than the threshold
    if n_diff_pixels < n_pixels_threshold: 
        print("frozen")
        overlay = frame.copy()
        cv2.putText(overlay,"freezing", placement, font, font_size, color, line_type)
        cv2.imshow("frame", overlay)
        diff_indices.append(1)      #update index as being a "frozen" frame
    else: diff_indices.append(0)    #update index as being a "non-frozen" frame
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

#analysis
#for i in range(len(diff_indices)):
#    pass