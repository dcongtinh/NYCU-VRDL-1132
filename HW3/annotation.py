import os
import glob
from tqdm.auto import tqdm
import numpy as np
import pycocotools
from detectron2.structures import BoxMode
from utils import *
from glob import glob
import cv2
import skimage.io as sio


def get_mask_annot(image_mask, category_id):
    binay_mask = np.asfortranarray(image_mask)
    seg = encode_mask(binay_mask)
    bbox_mode = BoxMode.XYWH_ABS
    bbox = pycocotools.mask.toBbox(seg).tolist()
    mask_annot = {
        'category_id': category_id,
        'segmentation': seg,
        'bbox_mode': bbox_mode,
        'bbox': bbox
    }

    return mask_annot


def generate_train_annot(file_root):
    filenames = glob(f"{file_root}/train/*")
    img_annots = []

    for image_id, folder_name in enumerate(tqdm(filenames)):
        image_path = f"{folder_name}/image.tif"
        masks_path = sorted(glob(f"{folder_name}/class*.tif"))

        image = cv2.imread(str(image_path))
        image_height, image_width, _ = image.shape

        mask_annots = []
        for image_mask in masks_path:
            category_id = int(image_mask.split('class')[-1][0])
            image_mask = sio.imread(image_mask)
            values_mask = np.unique(image_mask)
            for mask in values_mask:
                if mask > 0:
                    mask_annots.append(get_mask_annot(
                        image_mask == mask, category_id))

        annot = {
            'file_name': image_path,
            'height': image_height,
            'width': image_width,
            'image_id': image_id+1,
            'annotations': mask_annots
        }

        img_annots.append(annot)

    print('finish!')

    return img_annots


if __name__ == '__main__':
    file_root = 'nycu-hw3-data'
    img_annots = generate_train_annot(file_root)
    save_json(img_annots, os.path.join(file_root, 'train__annot.json'))
