from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
import cv2
import json
import copy
import numpy as np
import scipy
import json
import pickle
from lib.opts import opts
from lib.detector_gcn import Detector
import matplotlib.pyplot as plt
import re
import nibabel as nib

image_ext = ['jpg', 'jpeg', 'png', 'bmp', 'nii.gz']  # Add 'nii.gz' to image_ext
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    detector = Detector(opt)

    if os.path.isdir(opt.demo):
        input_files = []
        ls = os.listdir(opt.demo)
        ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 按最后一个数字升序排序

        for file_name in ls:
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext or file_name.endswith('.nii.gz'):  # Check for .nii.gz specifically
                input_files.append(os.path.join(opt.demo, file_name))
    else:
        input_files = [opt.demo]

    for input_file_path in input_files:
        out = None
        out_name = os.path.basename(input_file_path)
        print('Processing:', input_file_path)
        print('out_name', out_name)

        if opt.debug < 5:
            detector.pause = False
        cnt = 0
        results = {}
        all_slices_segmentations = []

        if input_file_path.endswith('.nii.gz'):
            nii_img = nib.load(input_file_path)
            nii_data = nii_img.get_fdata()
            affine = nii_img.affine
            header = nii_img.header

            # Iterate over slices (assuming 3D NIfTI, iterate along the last axis)
            for i in range(nii_data.shape[-1]):
                slice_data_raw = nii_data[..., i]
                slice_data_oriented = np.rot90(slice_data_raw, k=1)  # Apply rotation

                # Normalize and convert to 3-channel image if necessary
                # Normalization should be done on the oriented slice
                min_val = np.min(slice_data_oriented)
                max_val = np.max(slice_data_oriented)
                if max_val - min_val > 1e-5:  # Avoid division by zero or near-zero
                    slice_data_normalized = (slice_data_oriented - min_val) / (max_val - min_val) * 255
                else:
                    slice_data_normalized = np.zeros_like(slice_data_oriented)  # Handle flat images

                img = cv2.cvtColor(slice_data_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                ret = detector.run(img, [input_file_path], i)  # Pass slice index as cnt

                # Removed the block that saves individual .jpg slices for .nii.gz inputs
                # The final 3D volume will be saved as .nii.gz later

                time_str = 'frame {} |'.format(i)
                for stat in time_stats:
                    time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                print(time_str)
                results[i] = ret['results']
                # Assuming ret['generic'] contains the segmentation mask for the slice
                # This might need adjustment based on how your detector returns segmentations
                # Create a mask for the current slice from ret['results']
                # The mask should have the shape of the oriented slice
                current_slice_mask = np.zeros(slice_data_oriented.shape[:2],
                                              dtype=np.int16)  # Use int16 for class labels to support wider range
                if 'results' in ret and ret['results']:
                    # ret['results'] is expected to be {0: {cat_id: [detections]}}
                    # where 0 is the image_id for the single slice processed
                    processed_results = ret['results']

                    if isinstance(processed_results,
                                  list):  # Case 1: Tracking is ON, results is a list of detection dicts
                        for det_item in processed_results:
                            if isinstance(det_item, dict) and \
                                    'class' in det_item and \
                                    'poly' in det_item and \
                                    'score' in det_item and \
                                    det_item['score'] > opt.out_thresh:
                                try:
                                    cat_id = int(float(det_item['class']))  # class might be float
                                    # No specific cat_id range check here as int16 has a wider range
                                    label_to_draw = cat_id + 1  # Map 0-indexed class to 1-indexed label
 
                                    poly_coords = np.array(det_item['poly'])
                                    if not poly_coords.any():
                                        print(f"Warning: Skipping detection with empty polygon in slice {i}. Score: {det_item.get('score')}")
                                        continue
                                    scaled_poly_coords = poly_coords * opt.down_ratio
                                    poly_points_fill = scaled_poly_coords.astype(np.int32).reshape(-1, 1, 2)
                                    cv2.fillPoly(current_slice_mask, [poly_points_fill], label_to_draw)
                                except ValueError:
                                    print(
                                        f"Warning: Invalid class value '{det_item['class']}' for detection in slice {i}. Score: {det_item.get('score')}")
                                    continue
                                except Exception as e_loop:
                                    print(f"Error processing detection item {det_item} in slice {i}: {e_loop}")
                                    continue

                    elif isinstance(processed_results,
                                    dict):  # Case 2: Tracking is OFF, results is a dict {img_id: {cat_id: [dets]}}
                        detections_for_slice = processed_results.get(0, {})  # Assuming image_id for a single slice is 0
                        for cat_id_str, cat_detections_list in detections_for_slice.items():
                            try:
                                cat_id = int(float(cat_id_str))  # category_id might be float string
                                # No specific cat_id range check here as int16 has a wider range
                                label_to_draw = cat_id + 1  # Map 0-indexed class to 1-indexed label
 
                                for det in cat_detections_list:
                                    if isinstance(det, dict) and \
                                            'poly' in det and \
                                            'score' in det and \
                                            det['score'] > opt.out_thresh:
                                        poly_coords = np.array(det['poly'])
                                        if not poly_coords.any():
                                            print(f"Warning: Skipping detection in cat_id {cat_id} with empty polygon in slice {i}. Score: {det.get('score')}")
                                            continue
                                        scaled_poly_coords = poly_coords * opt.down_ratio
                                        poly_points_fill = scaled_poly_coords.astype(np.int32).reshape(-1, 1, 2)
                                        cv2.fillPoly(current_slice_mask, [poly_points_fill], label_to_draw)
                            except ValueError:
                                print(f"Warning: Could not convert cat_id '{cat_id_str}' to int for slice {i}.")
                                continue
                            except Exception as e_loop:
                                print(f"Error processing category {cat_id_str} detections in slice {i}: {e_loop}")
                                continue
                    else:
                        print(
                            f"Warning: ret['results'] is of unexpected type: {type(processed_results)} for slice {i}.")

                all_slices_segmentations.append(current_slice_mask)
                # The case for no detections or issues in ret['results'] will result in a zero mask being appended, which is intended.
                cnt += 1

            if all_slices_segmentations:
                # Stack all segmentation slices to form a 3D volume
                segmentation_volume_oriented = np.stack(all_slices_segmentations, axis=-1)
                # Rotate the volume back to the original orientation
                # The original rotation was np.rot90(slice_data_raw, k=1) for each slice
                # So, we need to rotate k=-1 (or k=3) on the slice plane (axes 0 and 1)
                segmentation_volume = np.rot90(segmentation_volume_oriented, k=-1, axes=(0, 1))
                # At this point, segmentation_volume is np.int16
                # Create a new NIfTI image for the segmentation using the np.int16 data directly.
                # nibabel will adjust the header's datatype information based on segmentation_volume.dtype upon saving.
                seg_nii_img = nib.Nifti1Image(segmentation_volume, affine, header)
                # Define output path for the segmentation
                seg_out_name = out_name  # Use the original filename
                if not seg_out_name.endswith('.nii.gz'):  # ensure .nii.gz extension if somehow missing
                    seg_out_name = os.path.splitext(seg_out_name)[0] + '.nii.gz'
                seg_save_path = os.path.join('exp/results/imgs/', seg_out_name)
                if opt.save_results:
                    # Ensure the directory exists before saving
                    os.makedirs(os.path.dirname(seg_save_path), exist_ok=True)
                    nib.save(seg_nii_img, seg_save_path)
                    print(f'Saved NIfTI segmentation to {seg_save_path}')
            save_and_exit(opt, out, results, out_name, is_nifti=True)

        else:  # Existing logic for JPG, PNG etc.
            while True:
                if cnt < len([input_file_path]):  # Process one image at a time from the outer loop
                    img = cv2.imread(input_file_path)
                    if img is None:
                        print(f"Error: Could not read image {input_file_path}")
                        break  # Break from while loop for this image
                else:
                    save_and_exit(opt, out, results, out_name)
                    break  # Break from while loop for this image

                ret = detector.run(img, input_files, cnt)  # Pass full input_files list and current index cnt

                time_str = 'frame {} |'.format(cnt)
                for stat in time_stats:
                    time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                print(time_str)

                results[cnt] = ret['results']  # Use cnt for non-NIfTI results key

                if opt.output_imgs and 'generic' in ret and isinstance(ret['generic'], np.ndarray):
                    # out_name is os.path.basename(input_file_path)
                    name_part = os.path.splitext(out_name)[0]  # Get filename without extension
                    save_file_name = '{}.jpg'.format(name_part)
                    save_path = os.path.join('exp/results/imgs/', save_file_name)
                    cv2.imwrite(save_path, ret['generic'])

                cnt += 1  # Increment counter for non-NIfTI images

                if cv2.waitKey(1) == 27:
                    save_and_exit(opt, out, results, out_name)
                    return  # Return from demo function
            # This save_and_exit is for the case where the loop finishes naturally for non-NIfTI
            save_and_exit(opt, out, results, out_name)


def save_and_exit(opt, out=None, results=None, out_name='', is_nifti=False):
    if not is_nifti and opt.save_results and (results is not None):
        # Adjust save_dir to be in the same directory as the input or a specified output directory
        # For example, save next to the input file or in a general 'results' folder
        base_dir = os.path.dirname(opt.demo) if os.path.isdir(opt.demo) else os.path.dirname(opt.demo)
        save_dir = os.path.join(base_dir,
                                '{}_results.json'.format(opt.input_mode + '_' + os.path.splitext(out_name)[0]))
        print('saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)),
                  open(save_dir, 'w'))
    if opt.output_imgs and out is not None:
        out.release()
    # Removed sys.exit(0) to allow processing of multiple files if opt.demo is a directory
    # sys.exit(0) # This will exit after the first file if opt.demo is a directory


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
