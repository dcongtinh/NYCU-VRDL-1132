import os
import argparse
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import detection_utils
from utils import *
from tqdm.auto import tqdm


def inference_step(model, test_img_ids):
    answer_list = []
    for test_image in tqdm(test_img_ids, total=len(test_img_ids)):
        image_path = os.path.join(
            'nycu-hw3-data', 'test', test_image['file_name'])
        image = detection_utils.read_image(image_path, format='BGR')
        test_image_id = test_image['id']
        pred = model(image)['instances'].to('cpu').get_fields()

        instances = convert_to_coco(test_image_id, pred)
        answer_list += instances

    return answer_list


def main(args):
    cur_time = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"

    cfg = get_cfg()
    cfg.merge_from_file(f'{args.checkpoint}/config.yaml')
    cfg.MODEL.WEIGHTS = os.path.join(
        args.checkpoint,
        args.weight_name)

    model = DefaultPredictor(cfg)
    test_img_ids = read_json(args.dataset)
    answer_list = inference_step(model, test_img_ids)
    save_root = args.checkpoint
    save_path = os.path.join(save_root, 'test-results.json')
    os.makedirs(save_root, exist_ok=True)
    save_json(answer_list, save_path)

    os.system(
        f"zip -j {save_root}/{cur_time}_{args.weight_name[:-4]}.zip {save_path}")
    print(f"Submission files saved: {save_root}/{cur_time}.zip")
    print('finish!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str,
                        default='tensorboard/mask_rcnn_R_50_DC5_3x/5class--Adagrad_epoch225_bs14_lr0.001_WarmupCosineLR_df0.1__cuda1')
    parser.add_argument('--weight_name', type=str, default='model_final.pth')
    parser.add_argument('--dataset', type=str,
                        default='nycu-hw3-data/test_image_name_to_ids.json')

    args = parser.parse_args()
    print(args)
    main(args)
