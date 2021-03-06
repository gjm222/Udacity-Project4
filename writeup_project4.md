
# Advanced Lane Finding Project (Project 4)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Explanation of Python Source Files
* `prog4-cal.py` -  The code does the camera calibration and saves the mtx and dist coefficients to and pickle file called "cal_pickle.p"
* `proj4.py` - The main code that loads the above pickle file and does the main functions.
* `proj4_video_gen.py` -  This is and altered version of `proj4.py` code that can run the code over a given video.
* `lane_locator.py` - This is class used by both `proj4.py` and `proj4_video_gen.py`

### Camera Calibration

#### 1. Computation of the Camera Matrix and Distortion Coefficients

The code for this step is contained in lines #15 through #47 of the file called `proj4-cal.py`).  
I start by preparing "object points", which will be the (x, y, z) coordinates of the 9x6 chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I then saved the mtx and dist coefficeints to a pickle file because this only needs to be done once.

### Pipeline (single images)

#### 1. Below is an example of a distortion-corrected image.

Here is the original test image...
![Original](./images/test1.jpg)

After loading the mtx and dist distortion coefficients from the camera calibration pickle file and applying `cv2.undistort()` function to the test image I obtained the following result:
![Un-distorted](./images/undistorted0.jpg)
Notice the white car to the right of the image to see the effect.  This was accomplished in lines #33 and #137 of the code in file `proj4.py` 
 
#### 2. Use of color transforms and gradients to create a thresholded binary image.  
I used a combination of the color space RGB and HSV thresholds to generate a binary image (thresholding function `color_thresh` starting at line 144 in `proj4_video_gen.py`).  I tried using gradients as well but did not get great results even when increasing the kernel size to 5x5 and 9x9 for the gradients which did smooth out the lines but also picked up noise.  The color thresholds focussing on yellow and white lines using a combination of RGB and HSV color spaces based on the yellow hue from HSV and the red and green values from RGB provided very distinct lines in the images.  Using the colors worked very well in shadows.  Here is a binary image example.... 

![Processed Binary](./images/bin3.jpg)

#### 3. Perspective Transformation

The code for my perspective transform appears in lines 366 through 385 in the file `proj4_video_gen.py`.  The code takes an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
    src = np.float32([[285,675], 
                     [1042,675],
                     [509,511],
                     [792,511]])
    
    dst = np.float32([[320, 720], 
                      [920, 720], 
                      [320, 500],
                      [920, 500]]) 

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 285, 675      | 320, 720     | 
| 1042, 675     | 920, 720     |
| 509, 511      | 320, 500     |
| 792, 511      | 920, 500     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.  

Before perspective transform...
![Before](./images/bin4.jpg)

After perspective transform...
![After](./images/warped4.jpg)


#### 4. Location of Lane-Line Pixels

I created a class called lane_line_finder in `lane_locator.py` to first do a histogram of the bottom half of the perspective binary image to find a starting location of the lanes at the bottom of the image. The class is instatiated on line 588 and called on line 389 of `proj4_video_gen.py`. The class code uses nine sliding windows (one for each vertical level of the perspective image) to locate the line in each of the levels of the image.  The windows slide based on the mean values of the pixels in the window.  A window for each lane is determined by first finding the center point and then using a given margin and the height of the image divided by the number of windows (in this case, 9).  If the initial left and right points were found prevously the histogram is not performed again unless a "reset" is determined to be necessary because of issues with the data. The right lane tends to have missing lines because they are usually dashed in the test data. Because of this bias, if the right lane windows has no pixels in the window, the right lane center point will "bump" in the same direction of the previous window. The code that does this is on lines 109 to 119 in `lane_locator.py`.  See detected lanes image below...

![Lanes Detected](./images/visual1.jpg)

Logic was added to determine if the lines were parallel and if they deviated much from the averay poly line coefficients.  The code is on lines 400 through 463 in `proj4_video_gen.py`.  A configurable amount of successful poly line coefficeints were saved so that averaging and comparing coulde be done.

To visually see the result, the x and y points found by the lane_locator class were fitted to numpy polylines from which a green colored trapezoidal polygon was created and is done on lines 487 through 497 of `proj4_video_gen.py`.  Finally, the perspective trapezoidal image was inverse transformed back to the original perspective and overlayed upon the original image.

The radius of curvature with a function starting on line 188 and position of the vehicle with respect to center was done in lines 547 through 548 in my code in `proj4_video_gen.py` and overlayed onto the final image using cv2.

Here is an example of my result on a test image:

![Image with Overlay](./images/final1.jpg)


### Pipeline (video) `marked_video.mp4`

Here's a [link to my video result](marked_video.mp4) 


### Discussion of Issues

#### General discussion

I mainly used several techniques that were done in the class lessons.  I played around with various gradients and found that using the highest thresholded intensity values (like 150 to 255) worked best but did not do well in shady regions.  Using the RGB and HSV color spaces proved to be the best so that is what i used exclusively for line detection.

At first I had difficulty transforming the perspective to get the lane lines without too much noise.  I am surprised at how much this effects the results of the entire project.  After transforming closer to the sides and stretching the length I got the desired result.  

#### Improvements

The code is somewhat biased towards using the left hand lane to determine what to do with the right hand lane if no data is present in the window. More work would need to be done to make it work well on roads without a solid left lane line marker.  I never did look in to using convolution to detect the lanes which may have resolved some issues.  I also should have done an averaging technique on lines in previous frames to make it more robust.

The code is heavily dependent upon color of the lines to detect lanes.  A more robust implementation would be necessary to accomodate a more diverse problem.

Also, the `proj4.py` and `proj4_video.gen.py` code should have been combined to avoid sychronization issues.


