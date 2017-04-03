# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:07:19 2017

@author: Greg McCluskey
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

#Calibrate camera and save results to pickle file. Only have to run this once 
objpoints = [] #Object points
imgpoints = [] #Image points

#Create reference coords            
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Get file names of calibration images
images = glob.glob('camera_cal\calibration*.jpg')

#For each image found... 
for index, fname in enumerate(images):
    
    #Read image
    img = mpimg.imread(fname)
    #Grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Find checkerboard points
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        outname = "output_images\chessboard" + str(index) + ".jpg"
        cv2.imwrite(outname, img)
        
#Get image info
img = cv2.imread('camera_cal\calibration1.jpg')     
img_shape = (img.shape[1], img.shape[0])   

#Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

#Save off calibration so we wont have to do it again
dpickle = {}
dpickle["mtx"] = mtx
dpickle["dist"] = dist
pickle.dump(dpickle, open('cal_pickle.p', 'wb'))       
        
