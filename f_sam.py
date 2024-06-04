from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# Create a FastSAM model
model = FastSAM('/home/ruveyda/Downloads/FastSAM-s.pt')  # or FastSAM-x.pt

from ultralytics import FastSAM

# Define an inference source
source = '/home/ruveyda/FastSAM/examples/dogs.jpg'

# Run inference on an image
everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# Prepare a Prompt Process object
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

# Everything prompt
ann = prompt_process.everything_prompt()

# Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

# Text prompt
ann = prompt_process.text_prompt(text='a photo of a dog')

# Point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
prompt_process.plot(annotations=ann, output='./')
