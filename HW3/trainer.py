import copy
import torch
import numpy as np
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.build import build_detection_train_loader
from detectron2.engine.defaults import DefaultTrainer
from detectron2.solver.build import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping


class Trainer(DefaultTrainer):  # SGD default
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=custom_mapper)

        return dataloader


class TrainerAdam(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=custom_mapper)

        return dataloader

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(model)
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY)


class TrainerAdamW(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=custom_mapper)

        return dataloader

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(model)
        return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY)


class TrainerAdagrad(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=custom_mapper)

        return dataloader

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(model)
        return maybe_add_gradient_clipping(cfg, torch.optim.Adagrad)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY)


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    img = utils.read_image(dataset_dict['file_name'], format='BGR')

    # Define augmentations
    augmentations = [
        T.RandomBrightness(0.9, 1.1),
        T.RandomCrop('relative', (0.5, 0.5)),
        T.ResizeShortestEdge(short_edge_length=608,
                             max_size=800, sample_style="choice"),
        T.RandomFlip(prob=0.5)
    ]

    # Apply augmentations
    aug_input = T.StandardAugInput(img)
    transforms = aug_input.apply_augmentations(augmentations)
    img = aug_input.image
    img_shape = img.shape[:2]

    # Convert image to tensor (C, H, W) format
    dataset_dict['image'] = torch.as_tensor(
        np.ascontiguousarray(img.transpose(2, 0, 1)))

    # Transform annotations
    annotations = dataset_dict.pop("annotations")

    transformed_annos = [
        utils.transform_instance_annotations(annotation, transforms, img_shape)
        for annotation in annotations
    ]

    # Convert annotations to instances
    instances = utils.annotations_to_instances(
        transformed_annos, img_shape, mask_format='bitmask'
    )

    # Update bounding boxes from masks
    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

    # Filter out empty instances
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict
