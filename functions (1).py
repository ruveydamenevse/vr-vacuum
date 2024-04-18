#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().system('pip install opencv-python')
import cv2 #OPEN CV
from matplotlib import pyplot as plt #to show the images


# In[2]:


def process_img(image): #basically tuning the parameters before getting the contours 
    #high pass fft, low frequence x 0 
    #sk image
    image_d = cv2.fastNlMeansDenoisingColored(image,None,19,19,11,30) #denoising
    Gray = cv2.cvtColor(image_d, cv2.COLOR_BGR2GRAY) # turn gray  
    blurred = cv2.GaussianBlur(Gray,(11,11),1)
    # Apply FFT to remove low-frequency components (high-pass filter)
    f = np.fft.fft2(blurred)
    fshift = np.fft.fftshift(f)
    rows, cols = blurred.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # Decrease the size of the area to zero out
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    Blur = np.uint8(img_back)
    Canny = cv2.Canny(Blur,150,150) 
    kernel = np.ones((3,3) , np.uint8)
    imgDial = cv2.dilate(Canny, kernel,iterations=5)
    return imgDial


# In[3]:


def paintFloor(img):

    img_processed = process_img(img)
    contours, hierarchy = cv2.findContours(img_processed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    height, width = img.shape[:2] # get matrix sizes
    
    # Create a mask for the convex hull
    mask = np.zeros_like(img)  
    result = img.copy() #we will work on result as to not alter original img
    
    all_coordinates = []
    #detect contour
    for i in contours: #look through contours
        area = cv2.contourArea(i) # get Area
        if area > 800:  # Area threshold
            # Calculate the centroid of the contour
            M = cv2.moments(i)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Check if the centroid lies above 1/3 of the image
                if cy < height * 2.4 / 3:
                    cv2.drawContours(result, [i], -1, (0, 255, 0), 2)  # Draw green contour lines
                    for point in i:
                        x, y = point[0]
                        # Append the (x, y) coordinate to the list
                        all_coordinates.append((x, y))
                
    all_coordinates = sorted(all_coordinates, key=lambda x: x[0])    
    
    max_y_values = {}
    for x, y in all_coordinates:
        if x not in max_y_values:
            max_y_values[x] = [y]
        else:
            max_y_values[x].append(y)
            max_y_values[x].sort(reverse=True)
            max_y_values[x] = max_y_values[x][:2]

    # Filter the coordinates to include only the top y-values for each x-coordinate
    filtered_coordinates = [(x, y) for x in max_y_values for y in max_y_values[x]]
    
    for c in filtered_coordinates:
        x, y = c
        mask[y:height,x] = (0,0,255)
        
    result = cv2.addWeighted(result, 1, mask, 0.3, 0)
      
    return result


# In[4]:


f1 = cv2.imread(r"red_rosin_taped_floors.jpg")


# In[5]:


c= paintFloor(f1)


# In[7]:





# In[ ]:





# In[ ]:




