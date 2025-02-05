import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import time 
import math
import random
# %config InlineBackend.figure_format = 'retina'
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# However, setting this parameter is not the recommended solution (as it says in the error message). Instead, you may try setting up your conda environment without using the Intel Math Kernel Library, by running:

# conda install nomkl --channel conda-forge
# (Note, the --channel conda-forge option tells conda to install the package from the conda-forge channel, which is a good idea generally, and may be necessary in this case)

# You may need to install some packages again after doing this, if the versions you had were based on MKL (though conda should just do this for you). If not, then you will need to do the following:

# install packages that would normally include MKL or depend on packages that include MKL, such as scipy, numpy, and pandas. Conda will install the non-MKL versions of these packages together with their dependencies (see here)

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.mkdir('trained-weights')

# # Download trained-weights from my release page or import your trained-weights and set path
# %cd trained-weights
# !wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/detr_r50_ep15.tar

# # Extract tar file
# !tar -xf detr_r50_ep15.tar
# !rm -r detr_r50_ep15.tar
# %cd ..
     
# Load model from torch.hub and load ckpt file into model

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2)
from models import detr
# model = detr() model_builder etc..
TRAINED_CKPT_PATH = 'outputs/models/sentinel2_finetune_20_epoch.pth'
# TRAINED_CKPT_PATH = 'outputs/eval/latest.pth'
checkpoint = torch.load(TRAINED_CKPT_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
 
CLASSES = ['n/a','ship']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741]]
     

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        # ax.text(xmin, ymin, text, fontsize=2,
        #         bbox=dict(facecolor='yellow', alpha=0.3))
    plt.axis('off')
    plt.show()

def postprocess_img(img_path): 
  im = Image.open(img_path)

  # mean-std normalize the input image (batch-size: 1)
  img = transform(im).unsqueeze(0)

  # propagate through the model
  start = time.time()
  outputs = model(img)
  end = time.time()
  print(f'Prediction time per image: {math.ceil(end - start)}s ', )

  # keep only predictions with 0.7+ confidence
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > 0.5

   # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  
  plot_results(im, probas[keep], bboxes_scaled)
     
# postprocess_img('datasets/sentinel2_coco/test/S2A_MSIL2A_20230306T022551_N0509_R046_T51RUM_20230306T065958_800_10400__png.rf.2a607c661fcc9fdd7664d6a5ba14c86f.jpg')
# Load test image paths

TEST_IMG_PATH = 'datasets/sentinel2_coco/test'

# img_format = {'jpg', 'png', 'jpeg'}
# paths = list()

# for obj in os.scandir(TEST_IMG_PATH):
# #   if obj.is_dir():
#     print("is fir")
#     # paths_temp = [obj.path for obj in os.scandir(obj.path) if obj.name.split(".")[-1] in img_format]
#     paths.extend(obj)

# print('Total number of test images: ', len(paths))
# random.shuffle(paths)
     

# for i in paths[1:10]:
#   postprocess_img(i)

img_format = {'jpg', 'png', 'jpeg'}
paths = list()

for obj in os.scandir(TEST_IMG_PATH):
    if obj.is_file() and obj.name.split(".")[-1] in img_format:
        paths.append(obj.path)

print('Total number of test images: ', len(paths))
random.shuffle(paths)

for i in paths[1:10]:
    postprocess_img(i)