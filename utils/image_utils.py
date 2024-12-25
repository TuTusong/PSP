import torch
import numpy as np
import cv2
from PIL import Image
def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def load_img(filepath):
    img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    img = img/255.
    return img

def mask_img(filepath):
    image=Image.open(filepath)
    out=image.convert("RGB")
    img=np.array(out)
    mask_final=cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    mask_final =cv2.cvtColor(mask_final[1], cv2.COLOR_RGB2GRAY)
    return mask_final


def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    # print(ps)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)