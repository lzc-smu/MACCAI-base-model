# import os
# import numpy as np
# import json
# import nibabel as nib
# from skimage import measure

# from .generic_dataset_nii import GenericDataset, parse_labels_from_json

# class Dataset2(GenericDataset):
#     default_resolution = [512, 512]
#     num_categories = 13
#     class_name = ["liver", "right-kidney", "spleen", "pancreas", "aorta", "ivc", "rag", "lag", "gallbladder", "esophagus", "stomach", "duodenum", "left kidney"]
#     _valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#     cat_ids = {v: i + 1 for i, v in enumerate(_valid_ids)}
#     max_objs = 128

#     def __init__(self, opt, split):
#         data_dir = opt.data_dir
#         image_dir = os.path.join(data_dir, 'imagesTr')
#         all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
#         np.random.seed(42)
#         idxs = np.arange(len(all_files))
#         np.random.shuffle(idxs)
#         split_idx = int(len(all_files) * 0.9)
#         train_files = [all_files[i] for i in idxs[:split_idx]]
#         val_files = [all_files[i] for i in idxs[split_idx:]]
#         used_files = train_files if split == 'train' else val_files

#         dataset_json = os.path.join(data_dir, 'dataset.json')
#         label_map = parse_labels_from_json(dataset_json)
#         valid_ids = list(label_map.values())

#         super().__init__(opt, split=split, used_files=used_files, label_map=label_map, valid_ids=valid_ids)

#         self.num_samples = len(self.slices)
#         print('Loaded {} {} samples'.format(split, self.num_samples))



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import copy

from .generic_dataset_nii import GenericDataset


class Dataset2(GenericDataset):
    default_resolution = [512, 512]
    num_categories = 13
    class_name = ["liver", "right-kidney", "spleen", "pancreas", "aorta", "ivc", "rag", "lag", "gallbladder", "esophagus", "stomach", "duodenum", "left kidney"]
    _valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    cat_ids = {v: i + 1 for i, v in enumerate(_valid_ids)}

    max_objs = 128

    def __init__(self, opt, split):
        data_dir = opt.data_dir
        if split == 'val':
            img_dir = os.path.join(data_dir, 'test/JPEGImages/')
            split = 'test'
            ann_path = os.path.join(
                data_dir, 'test',
                'annotations.json')
        else:
            img_dir = os.path.join(data_dir, 'train/JPEGImages/')
            ann_path = os.path.join(
                data_dir, 'train',
                'annotations.json').format(split)

        self.images = None
        # load image list and coco
        super(Dataset2, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                # newest format
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    cat_id = item['class'] - 1
                    category_id = self._valid_ids[cat_id]
                    bbox = item['bbox']
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(item['score']))
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_coco.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results_coco.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


