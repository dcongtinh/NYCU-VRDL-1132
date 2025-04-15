import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from util import transform
import json
import numpy as np


class DigitDataset(Dataset):
    def __init__(self, root, mode, is_train=False):
        self.root = root
        self.is_train = is_train

        with open(f"{self.root}/{mode}__annot.json", 'rb') as f:
            self.annot_dict = json.load(f)
        image_list = []
        for file_name in self.annot_dict.keys():
            self.annot_dict[file_name][0] = np.array(
                self.annot_dict[file_name][0])
            self.annot_dict[file_name][1] = np.array(
                self.annot_dict[file_name][1])
            image_list.append(f"{self.root}/{mode}/{file_name}")

        # with open(f"{self.root}/{mode}.json", 'rb') as f:
        #     data = json.load(f)
        # # print(len(data['images']))
        # # print(data['annotations'])
        # # print(data['categories'])
        # self.annot_dict, image_list = {}, []
        # annots = data['annotations']
        # for img in tqdm(data['images'], total=len(data['images'])):
        #     # print(img)
        #     labels = [annot['category_id']for annot in annots if annot['image_id'] == img['id']]
        #     boxes = []
        #     for annot in annots:
        #         if annot['image_id'] == img['id']:
        #             bbox = annot['bbox']
        #             # let bbox be [left, top, right, bottom]
        #             # right = left + width
        #             bbox[2] += bbox[0]
        #             # bottom = top + height
        #             bbox[3] += bbox[1]

        #             boxes.append(bbox)

        #     image_list.append(f"{self.root}/{mode}/{img['file_name']}")

        #     self.annot_dict[img['file_name']] = (labels, boxes)

        # with open(f"{self.root}/{mode}__annot.json", "w") as f:
        #     json.dump(self.annot_dict, f, indent=4)
        # # print(self.annot_dict.keys()) # 'images', 'annotations', 'categories'
        # # print(aa)

        self.images = image_list

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i])
        image = image.convert('RGB')
        img_name = os.path.basename(self.images[i])

        # Read objects in this image (bounding boxes, labels)
        # (n_objects), (n_objects, 4)
        (labels, boxes) = self.annot_dict[img_name]
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)

        # Apply transformations
        image, boxes, labels = transform(
            image, boxes, labels, is_train=self.is_train)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["img_name"] = img_name

        return image, target

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        return images, targets


def get_test_img(idx):
    img_filename = "nycu-hw2-data/test/{}.png".format(idx)

    image = Image.open(img_filename)
    image = image.convert('RGB')

    image, _, _ = transform(image)

    return image
