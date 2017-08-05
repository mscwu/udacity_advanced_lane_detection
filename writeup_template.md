# Advanced Lane Finding Project #

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

[image1]: ./output_images/hsv1.jpg "HSV 1"
[image2]: ./output_images/hsv2.jpg "HSV 2"
[image3]: ./output_images/hsv3.jpg "HSV 3"
[image4]: ./output_images/hsv4.jpg "HSV 4"
[image5]: ./output_images/hsv5.jpg "HSV 5"
[image6]: ./output_images/hsv6.jpg "HSV 6"
[image7]: ./output_images/hls1.jpg "HLS 1"
[image8]: ./output_images/hls2.jpg "HLS 2"
[image9]: ./output_images/hls3.jpg "HLS 3"
[image10]: ./output_images/hls4.jpg "HLS 4"
[image11]: ./output_images/hls5.jpg "HLS 5"
[image12]: ./output_images/hls6.jpg "HLS 6"
[image13]: ./output_images/bgr1.jpg "BGR 1"
[image14]: ./output_images/bgr2.jpg "BGR 2"
[image15]: ./output_images/bgr3.jpg "BGR 3"
[image16]: ./output_images/bgr4.jpg "BGR 4"
[image17]: ./output_images/bgr5.jpg "BGR 5"
[image18]: ./output_images/bgr6.jpg "BGR 6"
[image19]: ./output_images/undistort_image.jpg "Undistorted image"
[image20]: ./output_images/undistort_road_image.jpg "Undistorted road image"
[image21]: ./output_images/yellow_lane1.jpg "Yellow Lane 1"
[image22]: ./output_images/yellow_lane2.jpg "Yellow Lane 2"
[image23]: ./output_images/yellow_lane3.jpg "Yellow Lane 3"
[image24]: ./output_images/yellow_lane4.jpg "Yellow Lane 4"
[image25]: ./output_images/yellow_lane5.jpg "Yellow Lane 5"
[image26]: ./output_images/yellow_lane6.jpg "Yellow Lane 6"
[image27]: ./output_images/white_lane1.jpg "White Lane 1"
[image28]: ./output_images/white_lane2.jpg "White Lane 2"
[image29]: ./output_images/white_lane3.jpg "White Lane 3"
[image30]: ./output_images/white_lane4.jpg "White Lane 4"
[image31]: ./output_images/white_lane5.jpg "White Lane 5"
[image32]: ./output_images/white_lane6.jpg "White Lane 6"
[image33]: ./output_images/yellow_and_white1.jpg "Yellow and White Lane 1"
[image34]: ./output_images/yellow_and_white2.jpg "Yellow and White Lane 2"
[image35]: ./output_images/yellow_and_white3.jpg "Yellow and White Lane 3"
[image36]: ./output_images/yellow_and_white4.jpg "Yellow and White Lane 4"
[image37]: ./output_images/yellow_and_white5.jpg "Yellow and White Lane 5"
[image38]: ./output_images/yellow_and_white6.jpg "Yellow and White Lane 6"
[image39]: ./output_images/sobelx1.jpg "Sobel X 1"
[image40]: ./output_images/sobelx2.jpg "Sobel X 2"
[image41]: ./output_images/sobelx3.jpg "Sobel X 3"
[image42]: ./output_images/sobelx4.jpg "Sobel X 4"
[image43]: ./output_images/sobelx5.jpg "Sobel X 5"
[image44]: ./output_images/sobelx6.jpg "Sobel X 6"
[image45]: ./output_images/sobel_dir1.jpg "Sobel Direction 1"
[image46]: ./output_images/sobel_dir2.jpg "Sobel Direction 2"
[image47]: ./output_images/sobel_dir3.jpg "Sobel Direction 3"
[image48]: ./output_images/sobel_dir4.jpg "Sobel Direction 4"
[image49]: ./output_images/sobel_dir5.jpg "Sobel Direction 5"
[image50]: ./output_images/sobel_dir6.jpg "Sobel Direction 6"
[image51]: ./output_images/comb_grad1.jpg "Combined Gradient 1"
[image52]: ./output_images/comb_grad2.jpg "Combined Gradient 2"
[image53]: ./output_images/comb_grad3.jpg "Combined Gradient 3"
[image54]: ./output_images/comb_grad4.jpg "Combined Gradient 4"
[image55]: ./output_images/comb_grad5.jpg "Combined Gradient 5"
[image56]: ./output_images/comb_grad6.jpg "Combined Gradient 6"
[image57]: ./output_images/bin_img1.jpg "Binary 1"
[image58]: ./output_images/bin_img2.jpg "Binary 2"
[image59]: ./output_images/bin_img3.jpg "Binary 3"
[image60]: ./output_images/bin_img4.jpg "Binary 4"
[image61]: ./output_images/bin_img5.jpg "Binary 5"
[image62]: ./output_images/bin_img6.jpg "Binary 6"
[image63]: ./output_images/vertices.jpg "Vertices"
[image64]: ./output_images/warped.jpg "Warped"
[image65]: ./output_images/warped2.jpg "Warped2"
[image66]: ./output_images/region_road_image1.jpg "Region 1"
[image67]: ./output_images/region_road_image2.jpg "Region 2"
[image68]: ./output_images/region_road_image3.jpg "Region 3"
[image69]: ./output_images/region_road_image4.jpg "Region 4"
[image70]: ./output_images/region_road_image5.jpg "Region 5"
[image71]: ./output_images/region_road_image6.jpg "Region 6"
[image72]: ./output_images/lanes_image1.jpg "Lanes 1"
[image73]: ./output_images/lanes_image2.jpg "Lanes 2"
[image74]: ./output_images/lanes_image3.jpg "Lanes 3"
[image75]: ./output_images/lanes_image4.jpg "Lanes 4"
[image76]: ./output_images/lanes_image5.jpg "Lanes 5"
[image77]: ./output_images/lanes_image6.jpg "Lanes 6"
[image78]: ./output_images/warped_binary_road_image1.jpg "Warped Binary 1"
[image79]: ./output_images/warped_binary_road_image2.jpg "Warped Binary 2"
[image80]: ./output_images/warped_binary_road_image3.jpg "Warped Binary 3"
[image81]: ./output_images/warped_binary_road_image4.jpg "Warped Binary 4"
[image82]: ./output_images/warped_binary_road_image5.jpg "Warped Binary 5"
[image83]: ./output_images/warped_binary_road_image6.jpg "Warped Binary 6"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the [IPython notebook](https://github.com/mscwu/udacity_advanced_lane_detection/blob/master/advanced_lane_lines.ipynb).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image19]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here I applied the distortion correction to one of the test images like this one:
![alt text][image20]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  

In order to find the best color selection method, I started with investigating the images in different color space and separate channels. The idea of using color selection to detect lanes is based upon the fact that lanes consist of white and yellow color. However, it has been proven that the most common color space, i.e., RGB space, is not suitable for picking colors when the image is subject to various lighting conditions.  

First I took a look at HSV space.  

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

Yellow lane appears dark in the H channel and bright in S and V channel. Pay attention to image No.4. Even though part of the yellow lane, close to the bottom of the image, is in the shadow, we can still clearly tell it apart from the road, especially in H and S channel.  

RGB space.  
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]

In fact, RGB color space is doing well for the white lane. However, again, pay attention to image No.4 and 5. The shadow starts to appear among all the channels, covering the yellow lane marking. This is the reason why I decided not to use RGB color space yellow color selection.

HLS space.
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
The HLS space shows similar results as HSV space. However, it seems that just by properly thresholding the S channel we can get most of the job done.  

Based on my observation, I decided to use HLS space to choose the yellow lane and the RGB space to pick the white lane.  
The code to pick the yellow lane is:  
```python
def pick_yellow(img, lower_yellow, upper_yellow, return_binary=False):
    # Convert BGR to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)   

    # Threshold the HLS image to get only yellow colors
    mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask= mask)
    if return_binary:
        return mask
    else:
        return res
```
Here are the sample road images with yellow lanes picked.  
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
The next step is to pick white lanes. Now, for white lanes, we can just select pixels with high RBG values in the RGB space.  
```python
def pick_white(img, lower_white, upper_white, return_binary=False):
    # Threshold the BGR image to get only white colors
    mask = cv2.inRange(img, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask= mask)
    if return_binary:
        return mask
    else:
        return res
```
Here are the sample road images with white lanes picked.
![alt text][image27]
![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]
Next, combine white and yellow lane selection.  
```python
def pick_white_yellow(img, lower_yellow, upper_yellow, lower_white, upper_white, return_binary=False):
    white = pick_white(img, lower_white, upper_white, True)
    yellow = pick_yellow(img, lower_yellow, upper_yellow, True)
    color_mask = cv2.bitwise_or(white, yellow)
    res = cv2.bitwise_and(img, img, mask = color_mask)
    if return_binary:
        return color_mask
    else:
        return res
```
Here are the sample road images with yellow and white lanes picked.
![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]
![alt text][image37]
![alt text][image38]

In fact, just by properly thresholding the colors, I had successfully isolated most lane markings from the rest of the image. However, I wanted to see if using sobel operator gave me more information from the image.  
```python
def sobel_x_gradient(img, k_threshold, return_binary=False):
    r_channel = img[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=9) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= k_threshold[0]) & (scaled_sobelx <= k_threshold[1])] = 1
    
    res = cv2.bitwise_and(img, img, mask = sxbinary)
    if return_binary:
        return sxbinary
    else:
        return res
```
![alt text][image39]
![alt text][image40]
![alt text][image41]
![alt text][image42]
![alt text][image43]
![alt text][image44]

It seemed that for image 5 particularly, sobel operator gave more information about the lane to us.  
Now, on top of that, I applied a sobel gradient direction filter.  
```python
def dir_threshold(img, sobel_kernel=3, thresh=[.7, 1.3]):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r_channel = img[:,:,2]
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output
 ```
![alt text][image45]
![alt text][image46]
![alt text][image47]
![alt text][image48]
![alt text][image49]
![alt text][image50] 

It seemed like gradient direction detector didn't reveal much more information but the lanes were still discernable in the images. It might be a good idea to use "and" operator to combine the gradient magnitude and graident direction filters together.  
```python
def combined_gradient(img, k_threshold=[20, 255], ang_threshold=[0.9, 1.2], kernel_size=15):
    sobelx = sobel_x_gradient(img, k_threshold, True)
    grad_dir = dir_threshold(img, kernel_size, ang_threshold)
    binary_output =  np.zeros_like(sobelx)
    binary_output[(sobelx == 1) & (grad_dir == 1)] = 1
    
    return binary_output
```
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]
![alt text][image56]
There didn't seem to be much more information added but that was what I wanted. I wanted use the color selection as the main detector and use gradient selector as a helper to provide a little bit more information.  
Now, it is time to combine the color and gradient thresholds and produce the final results.  
![alt text][image57]
![alt text][image58]
![alt text][image59]
![alt text][image60]
![alt text][image61]
![alt text][image62]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`.  The `warp()` function takes as input an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
v1 = (187, img.shape[0])
v2 = (591, 450)
v3 = (688, 450)
v4 = (1122, img.shape[0])
# define a function to warp the image to change perspective to birds eye view
def warp(img):
    '''Compute perspective transformation M and its inverse and a warped image
    
    Keyword arguments:
    img -- input image
    '''
    img = np.copy(img)
    img_size = (img.shape[1], img.shape[0])
    # source points
    src = np.float32([v1, v2, v3, v4])
    # desination points
    dst = np.float32([[450, img.shape[0]], [450, 0], [img.shape[1]-450, 0], [img.shape[1]-450, img.shape[0]]])
    # get transformation matrix M
    M = cv2.getPerspectiveTransform(src, dst)
    # get inverse transformation matrix invM
    invM = cv2.getPerspectiveTransform(dst, src)
    # create warped image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return (M, invM, warped)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image63]
![alt text][image64]

I also drew the vertices on another image with curved lanes.
![alt text][image65]

Now that we can both create binary images and warped images, they can be combined into the pipeline.  
![alt text][image78]
![alt text][image79]
![alt text][image80]
![alt text][image81]
![alt text][image82]
![alt text][image83] 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find lanes and fit a polynomial, I used a method called sliding window search.  
This method contains following steps:  
    1.  Create a histogram of the bottom half of the image. Possible lanes will show up with high histogram values.  
    2.  Find the peak points of left and right image plane separately and use them as the starting point of the search.  
    3.  Divide the image from bottom up into a number of slices and create windows of width "margin" that slides left and right with in each slice.  
    4.  For each slice of image, if the number of nonzero pixels included in the window is higher than a threshold, the window is recentered and the centroid is added to a list that keeps track of all the centroids. This step is iterated from bottom of the image to the top of the image.  
    5.  When all the centroids have been collected, a second order polynomial fit is created.  
The code is as follows.  
```python
# define a sliding window function to find lanes
def sliding_window_search(binary_warped, nwindows, margin, minpix, visualize = False):
    '''Find lanes in a binary warped image
    
    Keyword arguments:
    binary_warped -- input binary image and already warped into top view
    nwindows -- number of sliding windows
    margin -- width of windows, +/- margin
    minpix -- minimum number of pixels found to recenter the window
    visualize -- boolean value to turn on visualization, for testing only
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    #nwindows = 12
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    #margin = 100
    # Set minimum number of pixels found to recenter window
    #minpix = 5
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if visualize:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit = np.polyfit(lefty, leftx, 2)
    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if visualize: 
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
    return ploty, left_fitx, right_fitx, out_img, lefty, leftx, righty, rightx
```
The final results are shown below, along with lanes warped back on to the original image.  
![alt text][image72]
![alt text][image73]
![alt text][image74]
![alt text][image75]
![alt text][image76]
![alt text][image77] 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Here is the code I used.  
```python
def process_lane_data(lane, img_shape, verbose=False):
    """Process left and right lane fitted data from binary image
    
    Return radius of curvature, offset from lane center, derivative of lanes at bottom max y
    
    keyword arguments:
    lane -- [ploty, left_fitx, right_fitx]
    img_shape -- [height, width]
    verbose -- debug control
    
    """
    ploty = lane[0]
    left_fitx = lane[1]
    right_fitx = lane[2]
    # evaluate curvature at the bottom of the image
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate derivatives for parallelism check
    left_dot = 2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1]  
    right_dot = 2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1]
    
    # Calculate the new radii of curvature in meters
    left_curverad = ((1 + left_dot**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + right_dot**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Compute left and right lane at bottom of image
    left_bot_x = left_fitx[np.argmax(ploty)]
    right_bot_x = right_fitx[np.argmax(ploty)]
    
    # Compute lane center
    lane_center = (right_bot_x + left_bot_x) / 2
    # Compute camera location, assuming camera is mounted at center of vehicle
    camera_x = img_shape[1] / 2
    # Compute lateral offset, if offset > 0, vehicle deviates to the right, otherwise deviates to the left
    offset = camera_x - lane_center
    # Convert to real world unit
    offset = offset*xm_per_pix
    
    # Print for debugging
    if verbose:
        print("Left Lane Radius: {0:.2f} m, Right Lane Radius: {1:.2f} m" .format(left_curverad, right_curverad))
        if offset < 0:
            print("Offset: {:.2f} m left".format(offset))
        else:
            print("Offset: {:.2f} m right".format(offset))
    
    return left_curverad, right_curverad, left_dot, right_dot, offset
 ```
The funciton `process_lane_data()` takes as input the lane data, image input and a verbose boolean for debuging. The first step is to proper scale the lane data from image space to world space. A decent estimation of lane width and length is required for good calculation of radius of curvature. Then a new polynomial fit is created. The radius of curvature is calculated based on the mathematical definition of radius of curvature of a curve.  
To calculate the offset, it was asssumed that the camera was installed at the center of the car and thus, the mid point of the image was where the car was centered. The lane center was defined as the center of left and right lane at the bottom of the image. Then the offset would be the difference between these two values.  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The image has been shown in previous sections.

#### 7. Validation of detected lanes.
It is important to validate the detected lanes. Even though I have tried to tune the parameters to make a robust algorithm, it was not guaranteed that detection would be successful for every frame. When a bad detection happens, the program should be able to tell and take measures accordingly.  
```python
# define a function to validate newly detected lanes
def lane_validated(ploty, left_fitx, right_fitx, left_curverad, right_curverad, left_dot, right_dot, verbose=False):
    flag = True
    # check radius of curvature
    if left_curverad / right_curverad > 2 or left_curverad / right_curverad < 0.5:
        flag = False
        if verbose:
            print("Radius ratio", left_curverad / right_curverad)
            print("Radius check failed")
        return flag
    # check lane width, 300 pixels < lane width < 400 pixels
    left_bot_x = left_fitx[np.argmax(ploty)]
    right_bot_x = right_fitx[np.argmax(ploty)]
    if right_bot_x - left_bot_x > 400 or right_bot_x - left_bot_x < 300:
        flag = False
        if verbose:
            print("Lane width", right_bot_x - left_bot_x, "pixels")
            print("Lane width check failed")
        return flag
    # check parallelism
    if np.absolute(left_dot / right_dot) > 10 or np.absolute(left_dot / right_dot) < 0.1:    
        flag = False
        if verbose:
            print("Derivative ratio", left_dot / right_dot)
            print("Parallelism check failed")
        return flag
    return flag
```
The code above shows how I checked the validation of detection. There are three conditions to meet:  
1.  The radius of left and right lanes should be close. I used a threshold of 2 times.
2.  The width of the lanes must be within a specific range.
3.  The lanes should be approximately parallel to each other. I checked the first order derivative of the equation.
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/rP_rDzRg_pc)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
