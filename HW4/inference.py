import argparse
from tqdm.auto import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os

from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
from net.model import PromptIR
import lightning.pytorch as pl
import torch.nn as nn
from datetime import datetime
from PIL import Image


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)


def test(net, dataset):
    cur_time = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"
    output_path = f"{testopt.ckpt_dir}/{testopt.ckpt_name}__out/"
    os.makedirs(output_path, exist_ok=True)

    testloader = DataLoader(dataset, batch_size=1,
                            pin_memory=True, shuffle=False, num_workers=0)

    with torch.no_grad():
        for ([degraded_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            restored = net(degrad_patch)
            save_image_tensor(restored, output_path +
                              degraded_name[0] + '.png')

    output_npz = f"{testopt.ckpt_dir}/pred.npz"

    # Initialize dictionary to hold image arrays
    images_dict = {}

    # Loop through all files in the folder
    for filename in os.listdir(output_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(output_path, filename)

            # Load image and convert to RGB
            image = Image.open(file_path).convert('RGB')
            img_array = np.array(image)

            # Rearrange to (3, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))

            # Add to dictionary
            images_dict[filename] = img_array

    # Save to .npz file
    np.savez(output_npz, **images_dict)
    print(f"Saved {len(images_dict)} images to {output_npz}")

    os.system(
        f"zip -j {testopt.ckpt_dir}/{testopt.ckpt_name}__{cur_time}.zip {output_npz}")
    print(
        f"Submission files saved: {testopt.ckpt_dir}/{testopt.ckpt_name}__{cur_time}.zip")
    print('finish!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--test_path', type=str,
                        default="data_split/Test/", help='save path of test images')
    parser.add_argument('--output_path', type=str,
                        default="output/", help='output save path')
    parser.add_argument('--ckpt_dir', type=str,
                        default="train_ckpt/", help='checkpoint save path')
    parser.add_argument('--ckpt_name', type=str, default="",
                        help='checkpoint save path')
    testopt = parser.parse_args()

    ckpt_path = f"{testopt.ckpt_dir}/" + testopt.ckpt_name
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    testset = TestSpecificDataset(testopt)
    test(net, testset)
