# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 08:05:44 2017

@author: SamKat2
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

class lane_line_finder():
    def __init__(self, nwindows, margin=100, minpix=50 ):
        self.recent_centers = [0.0] * nwindows
        
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        
        self.left_rects = []
        self.right_rects = []
        
        self.nonzeroy = np.empty((0))
        self.nonzerox = np.empty((0))
                
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        
    def make_lane_windows(self, binary_warped, idx):
        
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint    
        
        nwindows = self.nwindows                       
                        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()                
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
                

        margin = self.margin        
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        self.left_rects = []
        self.right_rects = []
        
        leftx_last_bump = 0
        
        # Step through the windows one by one
        for window in range(nwindows):
                
            leftx_previous = leftx_current 
            
                   
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            #Draw rectangle of window          
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            
            #Save rectangles of window          
            self.left_rects.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
            self.right_rects.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
                leftx_last_bump = leftx_current - leftx_previous
            else: #go with the momentum if no data
                leftx_current += leftx_last_bump
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))
            else:
                # Not enough data found for right side so use left side to help
                rightx_current += (leftx_current - leftx_previous)
                            
                
        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(left_lane_inds)
        self.right_lane_inds = np.concatenate(right_lane_inds)
        
        return
        
        
    
        
            
        
        