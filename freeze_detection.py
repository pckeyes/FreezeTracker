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
#freeze_threshold:      Decides how many consecutive "frozen" frames consititutes
#                       a true freeze event

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math

#read in first n frames of video
cap = cv2.VideoCapture("/Users/piperkeyes/Desktop/fc_test_notone.mp4")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
prev_frames = []
n_prev_frames = 3
for i in range(0,n_prev_frames):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    prev_frames.append(frame)
    
#lists to store freezing data
diff_indices = []
diff_bool = []
diff_frames = []

#threshold params
black_threshold = 20
n_pixels_threshold = 100
fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
freeze_threshold = fps/2 #set to .5 seconds

#text params
suffix = "_notone"
placement = (20,20)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
color = (255,255,255)
line_type = 2
count = 0

#read in the video frame by frame and decide whether mouse is not moving ("frozen")
for frame_n in range(n_frames - n_prev_frames):
    #print(count)
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    diff = cv2.absdiff(frame,prev_frames[0])
    cv2.imshow("frame", frame)
    cv2.imshow("subtraction", diff)
    if count == 0: cv2.waitKey(5000)    #pauses video so user has time to move windows

    
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
        #print("frozen")
        overlay = frame.copy()
        cv2.putText(overlay,"freezing", placement, font, font_size, color, line_type)
        cv2.imshow("frame", overlay)
        diff_bool.append(1)         #update index as being a "frozen" frame
        diff_indices.append(count)  #update which frame was "frozen"
    else: diff_bool.append(0)    #update index as being a "non-frozen" frame
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


#un-thresholded analysis
freeze_epochs = []
freeze_frames = 0
for i in range(len(diff_bool)):
    if diff_bool[i] == 1:
        if i == len(diff_bool)-1 and freeze_frames != 0:
            freeze_frames += 1
            freeze_epochs.append(freeze_frames)
        else: freeze_frames += 1
    else:
        if freeze_frames > 0: freeze_epochs.append(freeze_frames)
        freeze_frames = 0

#thresholded analysis
#remove freeze epochs of fewer consequtive freeze_frames than freeze_threshold
diff_indices_thresholded = []
start_index = 1
count = 0
for i in range(1,len(diff_indices)):
    if diff_indices[i] - diff_indices[i-1] == 1: count += 1
    else:
        if count > freeze_threshold:
            for frame in diff_indices[start_index:i]:
                diff_indices_thresholded.append(frame)
        start_index = i
        count = 0
freeze_epochs = np.array(freeze_epochs)
freeze_epochs_thresholded = freeze_epochs[freeze_epochs > freeze_threshold]

        
#plot freeze epochs un-thresholded
fig = plt.figure()
plt.eventplot(diff_indices,lineoffsets=0, linelengths=.1)
plt.xlabel("Frame")
plt.xticks([0,100,200,300,400,500,600,700,800,900])
plt.yticks([])
fig.set_size_inches(10,10)
fig.savefig("freezing_eventplot" + suffix + ".eps", transparent=True, format="eps", dpi=1000)

#plot freeze epochs thresholded
fig = plt.figure()
plt.eventplot(diff_indices_thresholded,lineoffsets=0, linelengths=.1)
plt.xlabel("Frame")
plt.xticks([0,100,200,300,400,500,600,700,800,900])
plt.yticks([])
fig.set_size_inches(10,10)
fig.savefig("freezing_eventplot_thresholded" + suffix + ".eps", transparent=True, format="eps", dpi=1000)

#convert data to dataframes
#un-thresholded
df_frames = pd.DataFrame(np.array(diff_indices))
df_frames.columns = ["Frame"]
df_epochs = pd.DataFrame(np.array(freeze_epochs))
df_epochs.columns = ["Duration in frames"]
#thresholded
df_frames_thresholded = pd.DataFrame(np.array(diff_indices_thresholded))
df_frames_thresholded.columns = ["Frame"]
df_epochs_thresholded = pd.DataFrame(np.array(freeze_epochs_thresholded))
df_epochs_thresholded.columns = ["Duration in frames"]

#write dataframes to csv
df_frames.to_csv("freeze_frames" + suffix + ".csv")
df_epochs.to_csv("freeze_epochs" + suffix + ".csv")
df_frames_thresholded.to_csv("freeze_frames_thresholded" + suffix + ".csv")
df_epochs_thresholded.to_csv("freeze_epochs_thresholded" + suffix + ".csv")
    
