#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().run_line_magic('run', 'functions.ipynb')


# In[9]:


get_ipython().run_line_magic('run', 'matching_fct.ipynb')


# In[10]:


img= cv2.imread("examples_pdr/9.jpeg")


# In[11]:


template= cv2.imread("ref.png")


# In[12]:


def CombineFloorVacuum (img, template):   
    img_copy = paintFloor(img)
    x,y,best_similarity = match_template(img, template)
    
    # Resize the window
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 600, 800) 
    
    if best_similarity>0.77:
        cv2.rectangle(img_copy, x, y, 255, 5) #draw vacuum
        
    cv2.imshow("result", img_copy)
    cv2.waitKey(0)


# In[13]:


#CombineFloorVacuum (img, template)


# In[7]:


#1 ve 5 7 8 9 ref1


# In[8]:


#6 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




