import os
import json
import yaml
from datetime import datetime
from detectron2 import model_zoo
from detectron2.config import get_cfg
import numpy as np
from pycocotools import mask as mask_utils
import skimage.io as sio


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array


def convert_to_coco(image_id, pred):
    instances = []

    boxes = pred['pred_boxes'].tensor.numpy()
    scores = pred['scores'].numpy()
    labels = pred['pred_classes'].numpy()
    masks = pred['pred_masks'].numpy()

    for box, score, label, mask in zip(boxes, scores, labels, masks):
        instances.append({
            'image_id': image_id,
            'bbox': box.tolist(),
            'score': float(score),
            'category_id': int(label),
            'segmentation': encode_mask(mask)
        })

    return instances


def read_json(json_path):
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    return data_dict


def save_json(save_list, save_path):
    with open(save_path, 'w') as f:
        json.dump(save_list, f, indent=4)

    return None


def save_config(cfg, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return None


def get_save_path(args):
    cur_time = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"
    save_path = f"{args.output_dir}/{args.model}/5class--{args.optimizer}_epoch{args.epochs}_bs{args.batch_size}_lr{args.base_lr}_{args.scheduler}__cuda{args.device[-1]}/{cur_time}"
    os.makedirs(save_path, exist_ok=True)

    return save_path


def set_cfg(args):
    cfg = get_cfg()

    # Set the model
    model_path = os.path.join(
        'COCO-InstanceSegmentation', f'{args.model}.yaml')
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    # class1 + class2 + class3 + class4 + background
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

    # if 'C4' in args.model:
    #     cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    # elif 'FPN' in args.model:
    #     cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]

    # Set the customer dataset
    cfg.DATASETS.TRAIN = ("Instance_Segmentation_DATA", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 6

    # Set the optimizer
    n_samples = 209
    iter_one_epoch = int(n_samples / args.batch_size)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.LR_SCHEDULER_NAME = args.scheduler
    cfg.SOLVER.MAX_ITER = args.epochs * iter_one_epoch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.BASE_LR_END = 1e-6
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 10000
    # int(0.2*cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.WARMUP_ITERS = 3 * iter_one_epoch
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = (args.epochs // 3) * iter_one_epoch

    # Set for the inference step
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.TEST.EVAL_PERIOD = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = 800
    cfg.TEST.AUG["ENABLED"] = True
    cfg.TEST.AUG.MIN_SIZES = (1500, 1600, 1700)

    cfg.MODEL.DEVICE = args.device
    cfg.OUTPUT_DIR = get_save_path(args)
    save_config(cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    return cfg
