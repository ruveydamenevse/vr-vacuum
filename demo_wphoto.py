import numpy as np
from time import sleep
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from libcamera import Transform
import cv2
from matching_fct import calculate_similarity, match_template
from combine_fcts import CombineFloorVacuum #needs a template in addition to frame
from segment import fetch_segmented_image
from masks import process_image_for_masks, filter_masks_by_size
from roboflow2 import detect_objects_in_image
from roboflow2 import draw_bounding_boxes

def reduce_size(image): #processing takes too much time maybe a fix?
    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized
#15-30 fps dene !

def select_floor(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR) #go from 4 channels to 3
    im_res = fetch_segmented_image(reduce_size(im))
    im=reduce_size(im)
    im_res = cv2.cvtColor(im_res, cv2.COLOR_BGRA2BGR) #go from 4 channels to 3
    masks = process_image_for_masks(im_res)
    min_size_masks = filter_masks_by_size(masks)
    
    full_mask = np.zeros_like(im_res)  
    height, width = full_mask.shape[:2] # get matrix sizes

    for mask in min_size_masks[0:round(len(min_size_masks)*0.75)]:
        binary_mask = mask.astype(int)
        for i in range(height) :
            for j in range(width):
                if binary_mask[i,j] == 1:
                    full_mask[i,j] = (0, 0, 255)
    # mask is processed
    full_mask[np.all(full_mask == (0, 0, 0), axis=-1)] = (0,255,0)
    full_mask[np.all(full_mask == (0, 0, 255), axis=-1)] = (0,0,0)      
                
    #np array olarak toparlamak sureyi duzeltebilir
    res = cv2.addWeighted(im, 1, full_mask, 0.5, 0)  
    coordinates = detect_objects_in_image(im)
    if coordinates is not None:
        res = draw_bounding_boxes(res, coordinates)
        
    numpy_horizontal = np.concatenate((res, res), axis=1)
    return numpy_horizontal




