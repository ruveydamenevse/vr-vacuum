#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:27:20 2024

@author: mehmet.arslan1
"""

import numpy as np
import cv2

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

# Görüntüleri yükle
img = cv2.imread('dene.png', 0)
template =cv2.imread('ref.png', 0)
img_height, img_width = img.shape[:2]
template_height, template_width = template.shape[:2]

# Şablonun ana görüntüye oranını hesapla
width_ratio = template_width / img_width
height_ratio = template_height / img_height

# En büyük oranı bul ve biraz küçültmek için 1'den küçük bir faktör kullan
scale_factor = max(width_ratio, height_ratio)
if scale_factor > 1:
    scale_factor = 1 / scale_factor  # Şablon büyükse, küçült
else:
    scale_factor = 1  # Şablon zaten uygun boyuttaysa, değiştirme

# Yeniden boyutlandırma faktörünü kullanarak şablonu yeniden boyutlandır
resized_template = cv2.resize(template, (0, 0), fx=scale_factor, fy=scale_factor)

print("Ana görüntü boyutları: ", img.shape)
print("Şablon görüntü boyutları: ", template.shape) 
h, w = template.shape[:2]

# Template matching için kullanılacak yöntemler
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]


resized_h, resized_w = resized_template.shape[:2]    

# Her bir yöntem için
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

    # Sağ alt köşenin koordinatlarını hesapla
    bottom_right = (best_location[0] + best_resized_w, best_location[1] + best_resized_h)
    cv2.rectangle(img, best_location, bottom_right, 255, 5)
    cv2.imshow(f'Match - Method: {method}, Scale: {best_scale}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # En iyi eşleşmenin sol üst ve sağ alt koordinatlarını yazdır
    print(f'Method: {method}, Best Scale: {best_scale}, Similarity: {best_similarity:.2f}, Top-left Location: {best_location}, Bottom-right Location: {bottom_right}')
