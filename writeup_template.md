## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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
[image78]: ./output_images/warped_binary_image1.jpg "Warped Binary 1"
[image79]: ./output_images/warped_binary_image2.jpg "Warped Binary 2"
[image80]: ./output_images/warped_binary_image3.jpg "Warped Binary 3"
[image81]: ./output_images/warped_binary_image4.jpg "Warped Binary 4"
[image82]: ./output_images/warped_binary_image5.jpg "Warped Binary 5"
[image83]: ./output_images/warped_binary_image6.jpg "Warped Binary 6"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

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

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
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
```
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
```
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
```
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
```
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
```
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
```
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

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
