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
from demo_wphoto import reduce_size, select_floor
from roboflow2 import detect_objects_in_image
picam2 = Picamera2()
picam2.framerate = 24 #can be changed moving forward
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
#allow warm up
sleep(0.1)

while True:
    im = picam2.capture_array()
    numpy_horizontal = select_floor(im)
    cv2.imshow("Camera", numpy_horizontal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()  # Stop the camera when done
cv2.destroyAllWindows()  # Close all OpenCV windows
