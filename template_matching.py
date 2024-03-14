import numpy as np
import cv2

def calculate_similarity(template, image, method, match_location, template_size):
    h, w = template_size
    x, y = match_location
    matched_region = image[y:y+h, x:x+w]
    if method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
        # Bu metodlar zaten normalize edilmiş bir skor verirler.
        similarity = cv2.matchTemplate(matched_region, template, method)[0][0]
    else:
        # Diğer yöntemler için, manuel olarak benzerlik hesaplanabilir.
        # Örneğin, norm cross correlation kullanılabilir.
        result = cv2.matchTemplate(matched_region, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        similarity = max_val
    return similarity

# Görüntüleri yükle
img = cv2.resize(cv2.imread('dene.png', 0), (0, 0), fx=0.8, fy=0.8)
template = cv2.resize(cv2.imread('ref.png', 0), (0, 0), fx=0.5, fy=0.5)
h, w = template.shape[:2]

# Template matching için kullanılacak yöntemler
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# Ölçekleri tanımla
scales = np.linspace(0.5, 2.0, 5, 10)

# Her bir yöntem için
for method in methods:
    img2 = img.copy()
    best_similarity = 0  # En iyi benzerlik için değişken
    for scale in scales:
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        resized_h, resized_w = resized_template.shape[:2]
        result = cv2.matchTemplate(img2, resized_template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc
        # En iyi eşleşme ve benzerliği güncelle
        similarity = calculate_similarity(resized_template, img2, method, location, (resized_h, resized_w))
        if similarity > best_similarity:
            best_similarity = similarity
            best_location = location
            best_scale = scale

    # En iyi eşleşmeyi görselleştir
    bottom_right = (int(best_location[0] + resized_w), int(best_location[1] + resized_h))
    cv2.rectangle(img2, best_location, bottom_right, 255, 5)
    print(f'Method: {method}, Best Scale: {best_scale}, Similarity: {best_similarity:.2f}')
    cv2.imshow(f'Match - Method: {method}, Scale: {best_scale}', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
