from tempfile import TemporaryDirectory
from pathlib import Path
import os

from loguru import logger
from PIL import Image
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import fire
import numpy as np
import cv2
import numpy as np

def fetch_segmented_image(
    image: Image,
    # TODO: inject from config
    device='cpu',
    retina_masks=True,
    imgsz=1024,
    conf=0.4,
    iou=0.9,
) -> Image:

    model = FastSAM('/home/ruveyda/Downloads/FastSAM-s.pt')  # or FastSAM-x.pt

    
    # Run inference on image
    everything_results = model(
        image,
        device=device,
        retina_masks=retina_masks,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    # Prepare a Prompt Process object
    prompt_process = FastSAMPrompt(image, everything_results, device='cpu')

    # Everything prompt
    ann = prompt_process.everything_prompt()

    # TODO: support other modes:

    # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    #ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

    # Text prompt
    #ann = prompt_process.text_prompt(text='floor')

    # Point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    #ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])

    assert len(ann) == 1, len(ann)
    ann_item = ann[0]

    # hide original image
    ann_item.orig_img = np.ones(ann_item.orig_img.shape)

    with TemporaryDirectory() as tmp_dir:
        # Force the output format to PNG to prevent JPEG compression artefacts
        ann_item.path = ann_item.path.replace('.jpg', '.png')
        prompt_process.plot(
            [ann_item],
            tmp_dir,
            with_contours=False,
            retina=False,
        )
        result_name = os.path.basename(ann_item.path).replace('.jpg', '.png')
        logger.info(f"{ann_item.path=}")
        image_path = Path(tmp_dir) / result_name
        image = Image.open(image_path)
        os.remove(image_path)
        
    open_cv_image =np.array(image)
    
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    return open_cv_image

