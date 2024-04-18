#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv2


# In[2]:


def calculate_similarity(template, image, method, match_location, template_size):
    h, w = template_size
    x, y = match_location
    matched_region = image[y:y+h, x:x+w]
    if method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
        similarity = cv2.matchTemplate(matched_region, template, method)[0][0]
    else:
        result = cv2.matchTemplate(matched_region, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        similarity = max_val
    return similarity


# In[3]:


def match_template(img, template):
        methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] #methods we will use
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turn gray 
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) 
        
        img_height, img_width = img.shape[:2] # get shapes and find scale factor
        template_height, template_width = template.shape[:2]
        width_ratio = template_width / img_width 
        height_ratio = template_height / img_height
        
        scale_factor = max(width_ratio, height_ratio) #find scaling factor
        if scale_factor > 1:
            scale_factor = 1 / scale_factor  # Şablon büyükse, küçült
        else:
            scale_factor = 1  # Şablon zaten uygun boyuttaysa, değiştirme
        
        resized_template = cv2.resize(template, (0, 0), fx=scale_factor, fy=scale_factor) #resize template according to scale
        resized_h, resized_w = resized_template.shape[:2]   
        
        for method in methods:
            best_similarity = 0
            best_location = (0, 0)
            best_scale = 1
            best_resized_h = 0
            best_resized_w = 0
            result = cv2.matchTemplate(img, resized_template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                location = min_loc
            else:
                location = max_loc
            similarity = calculate_similarity(resized_template, img, method, location, (resized_h, resized_w))
            if similarity > best_similarity:
                best_similarity = similarity
                best_location = location
                best_scale = scale_factor
                best_resized_h = resized_h
                best_resized_w = resized_w
                
        bottom_right = (best_location[0] + best_resized_w, best_location[1] + best_resized_h)  
        
        return best_location, bottom_right, best_similarity


# In[4]:


#img= cv2.imread("dene.png")


# In[5]:


#template= cv2.imread("ref.png")


# In[6]:


#x,y = match_template(img, template)


# In[7]:


#cv2.rectangle(img, x, y, 255, 5)


# In[8]:


#cv2.imshow("res", img)
#cv2.waitKey(0)


# In[ ]:




