# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:07:19 2017

@author: SamKat2
"""

from moviepy.editor import VideoFileClip

import numpy as np
import cv2
import pickle
import glob
from lane_locator import lane_line_finder

bOutputVisual = True



# Read in the saved objpoints and imgpoints of already performed camera calibration
dist_pickle = pickle.load( open( "cal_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


# returns the undistorted image
def cal_undistort(img, mtx, dist):
    # Use cv2.calibrateCamera() and cv2.undistort()
    # undist = np.copy(img)  # Delete this line
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Apply threshold
    return grad_binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
                  
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    
    # Apply threshold
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


# Edit this function to create your own pipeline.
def color_thresh(img, s_thresh=(0, 255), v_thresh=(0, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel and threshold
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Convert to HSV color space and separate V channel and threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)    
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
      
    # Combine
    color_binary = np.zeros_like(v_channel)
    color_binary[(s_binary == 1) & (v_binary == 1)] = 1
    
    return color_binary


#Process incomming image in to final image with lane overay
def process_image(org_img):
    
    #Undistort
    img = cal_undistort(org_img, mtx, dist)
    
    
    #process image to get lines
    ksize = 3
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(150, 200)) #20 100
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(150, 200)) #20 100
    
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(50, 255))#30 100
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.6, 1.4)) #np.pi/2  
    
    color_binary = color_thresh(img, s_thresh=(150, 255), v_thresh=(128, 255))
    
    #Combine                          
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 1  
    combined = combined * 255        
                          
     
         
    #Coordinates obtained by painstakingly ;) eyeballing points off a straight
    #road image of the same resolution using MS paint pgm    
    src = np.float32([[610,440], 
                     [670,440],
                     [1042,675],
                     [285,675]])
    
    xoffset = 250 #x offset for dst points                          
    yoffset = -100 #y offset for dst points
    img_size = (img.shape[1], img.shape[0])    
    
    #Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[xoffset, yoffset], 
                      [img_size[0]-xoffset, yoffset], 
                      [img_size[0]-xoffset, img_size[1]], 
                      [xoffset, img_size[1]]])
        
    
    #use cv2.getPerspectiveTransform() to get M and inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    #use cv2.warpPerspective() to warp your image to a top-down view
    warped =  cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)
    
    
        
    #locate lanes
    lfinder.make_lane_windows(warped, 1)
    
    
    # Extract left and right line pixel positions
    leftx = lfinder.nonzerox[lfinder.left_lane_inds]
    lefty = lfinder.nonzeroy[lfinder.left_lane_inds] 
    rightx = lfinder.nonzerox[lfinder.right_lane_inds]
    righty = lfinder.nonzeroy[lfinder.right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)   
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create pixel points for left lane
    #pts = [[x,y] for x,y in zip(left_fitx, ploty)]
    #pts = np.array(pts, np.int32)        
    
    # Create pixel points for right lane
    #pts = [[x,y] for x,y in zip(right_fitx, ploty)]
    #pts = np.array(pts, np.int32)        
            
    
    #Create overlay image
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    
    
    #################################################
    
    
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient

    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                  for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])
    
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    y_eval = np.max(ploty)
    
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    
    
    #camera center
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix                 
    
    
    cv2.putText(result, 'Radius of Curvature = '+str(np.round(left_curverad,4))+'(m)', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Center Position = '+str(np.round(center_diff,4))+'(m)', (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
    #plt.imshow(result)
    
    
    return result


print("Starting...")

bfirst = True # For testing only 

#Initialize lane finder object
lfinder = lane_line_finder(9, 75, 50)
    
output_video = 'marked_video.mp4'
#output_video = 'marked_challenge_video.mp4'
#input_video = 'challenge_video.mp4'
input_video = 'project_video.mp4'

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

print("Done")    
   
    