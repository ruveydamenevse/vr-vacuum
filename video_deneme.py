#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', 'combine_fcts.ipynb')


# In[ ]:


template= cv2.imread("examples_pdr/ref_new.PNG")


# In[ ]:


import cv2
import numpy as np
 
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('examples_pdr/10.mp4')
 
# Loop until the end of the video
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)
 
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    x,y,best_similarity = match_template(frame, template)
        
    # using cv2.Gaussianblur() method to blur the video
    frame = paintFloor(frame)
    # (5, 5) is the kernel size for blurring.

    if best_similarity>0.77:
        cv2.rectangle(frame, x, y, 255, 5) #draw vacuum
    
        
    cv2.imshow('gblur', frame_new)
 
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
# release the video capture object
cap.release()
 
# Closes all the windows currently opened.
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




