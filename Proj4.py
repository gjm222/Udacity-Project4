# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:07:19 2017

@author: SamKat2
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from lane_locator import lane_line_finder
from line import lines



bOutputVisual = True





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
def color_thresh_org(img, s_thresh=(0, 255), v_thresh=(0, 255)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    #l_channel = hls[:,:,1]
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
             
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    rg_binary = np.zeros_like(r_channel)
    rg_binary[((r_channel > 250) & (g_channel > 250) & (b_channel < 5)) | 
            ((r_channel > 240) & (g_channel > 240) & (b_channel > 240))] = 1         
      
    
    # Threshold color channel
    color_binary = np.zeros_like(v_channel)
    color_binary[(s_binary == 1) & (v_binary == 1) | rg_binary == 1] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    return color_binary

# Edit this function to create your own pipeline.
def color_thresh(img, h_thresh=(20, 50), v_thresh=(0, 255), s_thresh=(0,255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel and threshold
    #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    s_channel = hls[:,:,1]
    print("schannel", np.amax(s_channel))
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    #hue         
    h_channel = hls[:,:,0]
    print("hchannel", np.amax(h_channel))
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1         
    
    # Convert to HSV color space and separate V channel and threshold
    #hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)    
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
             
    #Look for yellow and white             
    r_channel = img[:,:,2]
    g_channel = img[:,:,1]
    b_channel = img[:,:,0]
    rg_binary = np.zeros_like(r_channel)
    #rg_binary[((r_channel > 100) & (g_channel > 100)) | 
    #        ((r_channel > 220) & (g_channel > 220) & (b_channel > 220))] = 1         
    
            
    #identifies yellow line        
    #rg_binary[(r_channel > 200) & (g_channel > 150) ] = 1                 
    rg_binary[(r_channel > 200) & (g_channel > 150) ] = 1                 
      
               
               
    # Threshold color channel
    color_binary = np.zeros_like(v_channel)
    #color_binary[(s_binary == 1) & (v_binary == 1) | rg_binary == 1] = 1         
    #color_binary[((h_binary == 1 ) & (s_binary == 1)) | (rg_binary == 1)] = 1                      
    
    color_binary[(h_binary == 1) | ((r_channel > 200) & (g_channel > 150)) ] = 1                      
    
    return color_binary


def get_curve_rad(left_fit, right_fit, xm_per_pix, ym_per_pix):
    
    # Generate y values to plug in to make a line (0 - 719)
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    
    #Create right and left lines using pixel coefficients
    leftx = np.array([left_fit[0]*(y**2) + left_fit[1]*y + left_fit[2] for y in ploty])
    rightx = np.array([right_fit[0]*(y**2) + right_fit[1]*y + right_fit[2] for y in ploty])
    
    #Use max y value to calculate curvature 
    y_eval = np.max(ploty)
     
    # Fit new polynomials to x,y in world space in meters
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Return curvature in meters
    return left_curverad, right_curverad


#Check average line - 
def check_line_avg(fit, framesq, linedata, boverride=False):
    
    breturn = False
    print(len(framesq))
    if len(framesq) > int(maxframes / 2) and not boverride:
        
        buffer2 = 2.0
        buffer1 = 3.0
        buffer0 = 150.0
        
        '''buffer2 = 9999.0
        buffer1 = 9999.0
        buffer0 = 9999.0'''
        
        x2 = 0.0
        x1 = 0.0
        x0 = 0.0
        
        #Get average of coefficients
        for i in framesq:
            print('qqqq entry', i.current_fit)
            x2 += i.current_fit[0]
            x1 += i.current_fit[1]
            x0 +=  i.current_fit[2]
            #
        x2 = x2 / len(framesq)    
        x1 = x1 / len(framesq)    
        x0 = x0 / len(framesq)    
        if fit[0] > (x2 - buffer2) and fit[0] < (x2 + buffer2):
            if fit[1] > (x1 - buffer1) and fit[1] < (x1 + buffer1):
                if fit[0] > (x0 - buffer0) and fit[0] < (x0 + buffer0):
                    #if passed then add to frames queue
                    framesq.append(linedata)
                    if len(framesq)  > maxframes:
                        framesq.pop(0)                        
                        breturn = True
                        print("Passed!!!")
    else:
        
        if boverride:
            print("Overridden")
        else:
            print("Filling data")
        breturn = True
        framesq.append(linedata)
        if len(framesq)  > maxframes:
            framesq.pop(0)
    
    return breturn

#Check line parallel - Check coefficients of X2 an X1
def is_line_parallel(left_fit, right_fit, x2buffer=.3, x1buffer=0.5):
    
    breturn = False
    
    if (right_fit[0] > (left_fit[0] - x2buffer)) and (right_fit[0] < (left_fit[0] + x2buffer)):
        if (right_fit[1] > (left_fit[1] - x1buffer)) and (right_fit[1] < (left_fit[1] + x1buffer)):
            breturn = True

    return breturn


#Main functions

print("Starting...")

#Setup frames liw5
maxframes = 2
maxfails = 2
lframesq = []  
rframesq = [] 
lfailcount = 0
rfailcount = 0

 


#Calibrate camera


# Read in the saved objpoints and imgpoints of previous calibration
dist_pickle = pickle.load( open( "cal_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


# Get file names of test images
#images = glob.glob('test_images\\test*.jpg')
images = glob.glob('snapshots\\vlcsnap-*.jpg')


bfirst = True

#Initialize lane finder object
lfinder = lane_line_finder(9, 75, 50)

breset = True

#For each image found... 
for index, fname in enumerate(images):
    #Read original image
    org_img = cv2.imread(fname)
    #Undistort
    img = cal_undistort(org_img, mtx, dist)
    
    #output undistorted image
    outname = 'output_images\\undistorted' + str(index) + '.jpg'
    cv2.imwrite(outname, img)
    
    #process image to get lines
    ksize = 3
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(150, 200)) #20 100
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(150, 200)) #20 100
    
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(50, 100))#30 100
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3)) #np.pi/2  
    
    #color_binary = color_thresh(img, s_thresh=(150, 255), v_thresh=(128, 255))
    color_binary = color_thresh(img)
    combined = color_binary * 255
    
    #plt.imshow(img)
    
    #combine                          
    '''combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1) & (mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 1  
    combined = combined * 255
    '''
                          
    #output binary image
    outname = 'output_images\\bin' + str(index) + '.jpg'
    cv2.imwrite(outname, combined) 
     
    
    
    
    src = np.float32([
                     [285,675],
                     [1042,675], 
                     [509,511],
                     [792,511]
                     ])
    
    xoffset = 500 #x offset for dst points                          
    yoffset = 0 #y offset for dst points
    img_size = (img.shape[1], img.shape[0])    
    
    print("img_size[0]-xoffset", img_size[0]-xoffset)
    print("img_size[1]", img_size[1])
    
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[320, 720], 
                      [920, 720], 
                      [320, 550],
                      [920, 550]])
                      
        
    #Source coordinates obtained by eyeballing points off a straight
    #road image of the same resolution using MS paint pgm    
    '''src = np.float32([[610,440], 
                     [670,440],
                     [1042,675],
                     [285,675]])
    
    xoffset = 250 #x offset for dst points                          
    yoffset = 0 #y offset for dst points
    img_size = (img.shape[1], img.shape[0])    
    
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[xoffset, yoffset], 
                      [img_size[0]-xoffset, yoffset], 
                      [img_size[0]-xoffset, img_size[1]], 
                      [xoffset, img_size[1]]])
    '''    
    
    #use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    #use cv2.warpPerspective() to warp your image to a top-down view
    warped =  cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)
    
     #Output visual
    if bOutputVisual:
        #output binary image
        outname = 'output_images\\warped' + str(index) + '.jpg'
        cv2.imwrite(outname, warped) 
     
        
        
    #locate lanes
    lfinder.make_lane_windows(warped, index, breset)
    
    
    # Extract left and right line pixel positions
    leftx = lfinder.nonzerox[lfinder.left_lane_inds]
    lefty = lfinder.nonzeroy[lfinder.left_lane_inds] 
    rightx = lfinder.nonzerox[lfinder.right_lane_inds]
    righty = lfinder.nonzeroy[lfinder.right_lane_inds] 
    
    #################################################
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    
        
    
    # Fit a second order polynomial to each
    if len(leftx) > 1000:        
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = lframesq[-1].current_fit #Use latest    
        print("No left data")
        breset = True
        
    if len(rightx) != 0:    
        right_fit = np.polyfit(righty, rightx, 2)   
    else:    
        right_fit = rframesq[-1].current_fit #Use latest
        print("No right data")



    if not is_line_parallel(left_fit, right_fit) and len(rframesq) != 0:
        right_fit = rframesq[-1].current_fit #Use latest
        print("-----Failed parallel test-----")
    else: #Check line average   

        #set left line
        llinedata = lines()    
        llinedata.current_fit = left_fit
        #set right line
        rlinedata = lines()
        rlinedata.current_fit = right_fit    
        
        #Check for runaway fails
        if lfailcount > maxfails:
            bloverride = True
        else:
            bloverride = False
            
        if rfailcount > maxfails:
            broverride = True
        else:
            broverride = False    
        
        #Check left lane line
        if not check_line_avg(left_fit, lframesq, llinedata, bloverride):
            left_fit = lframesq[-1].current_fit
            lfailcount += 1
        else:
            lfailcount = 0 #reset
        
        #Check right lane line
        if not check_line_avg(right_fit, rframesq, rlinedata, broverride):
            right_fit = rframesq[-1].current_fit
            rfailcount += 1
        else:
            rfailcount = 0 #reset
        

    
    print("left_fit=", left_fit)
    print("right_fit=", right_fit)
    
    
    
    #Get curve radius
    left_curverad, right_curverad = get_curve_rad(left_fit, right_fit, xm_per_pix, ym_per_pix)
    print("Left curvature", left_curverad)
    print("Right curvature", right_curverad)
    
        
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Color the left and right lane marker lines
    
    # Create pixel points for left lane
    pts = [[x,y] for x,y in zip(left_fitx, ploty)]
    pts = np.array(pts, np.int32)        
    left_lane_pts = pts.reshape((-1,1,2))
    
    # Create pixel points for right lane
    pts = [[x,y] for x,y in zip(right_fitx, ploty)]
    pts = np.array(pts, np.int32)        
    right_lane_pts = pts.reshape((-1,1,2))
    
    #Output visual
    if bOutputVisual:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped, warped, warped))*255
        
        out_img[lfinder.nonzeroy[lfinder.left_lane_inds], lfinder.nonzerox[lfinder.left_lane_inds]] = [255, 0, 0]
        out_img[lfinder.nonzeroy[lfinder.right_lane_inds], lfinder.nonzerox[lfinder.right_lane_inds]] = [0, 0, 255]
        
        # Draw rectangles
        
        for corner in lfinder.left_rects:
            print("corner\=",corner[0])
            cv2.rectangle(out_img,corner[0],corner[1],(0,255,0), 2) 
        for corner in lfinder.right_rects:
            cv2.rectangle(out_img,corner[0],corner[1],(0,255,0), 2) 
        
        # Draw left line marker
        cv2.polylines(out_img,[left_lane_pts],False,(0,255,0),5)
        
        # Draw right line marker
        cv2.polylines(out_img,[right_lane_pts],False,(0,255,0),5)
              
        #output marked up lanes
        outname = 'output_images\\visual' + str(index) + '.jpg'
        cv2.imwrite(outname, out_img) 
        
    
    
    
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
    
    
    
    
    
    #camera center
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix                 
    
    
    cv2.putText(result, 'Radius of Curvature = '+str(np.round(left_curverad,4))+'(m)', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Center Position = '+str(np.round(center_diff,4))+'(m)', (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
    #plt.imshow(result)
    
    #output marked up lanes
    outname = 'output_images\\final' + str(index) + '.jpg'
    cv2.imwrite(outname, result) 
    
    '''
    if bOutputVisual:
        #output marked up lanes
        outname = 'output_images\\markedlane' + str(index) + '.jpg'
        cv2.imwrite(outname, ovl_img) 
    
    #Unwarped lane locators
    unwarped_ovl_img =  cv2.warpPerspective(ovl_img, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    combo = cv2.addWeighted(img, 1.0, unwarped_ovl_img, 1.0, 0.0)
    
    if bOutputVisual:
        #output marked up lanes
        outname = 'output_images\\final' + str(index) + '.jpg'
        cv2.imwrite(outname, combo) 
    
    '''
    bfirst = False
    print('')
    print('=========================================')
    
print("Done")    
   
    
