#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import nibabel as nib
from PIL import Image
import cv2
from tqdm import tqdm
import datetime
from skimage import measure # Added import
from shapely.geometry import Polygon, MultiPolygon # Added import

def nii_to_jpg_slices(nii_path, output_dir, base_filename, is_mask=False):
    """
    Converts all slices of a NIfTI image to JPG format.
    Normalizes and converts to 8-bit for image, direct conversion for mask.
    Rotates image slices to correct orientation.
    Returns a list of (jpg_filepath, width, height) for each saved slice.
    """
    print(f"[nii_to_jpg_slices] Starting to process {nii_path}")
    saved_slices_info = []
    try:
        print(f"    Attempting to load NIfTI image: {nii_path}")
        nii_img = nib.load(nii_path)
        print(f"    Successfully loaded NIfTI image: {nii_path}")
        img_data_full = nii_img.get_fdata()
        print(f"[nii_to_jpg_slices] Loaded NIfTI image: {nii_path}. Shape: {img_data_full.shape}, Type: {img_data_full.dtype}.")

        if img_data_full.ndim not in [3, 4]:
            print(f"Unsupported NIfTI dimension: {img_data_full.ndim} for {nii_path}. Expected 3D or 4D.")
            return saved_slices_info
        
        if img_data_full.shape[2] == 0:
            print(f"NIfTI file {nii_path} has 0 slices. Skipping this file.")
            return saved_slices_info

        num_original_slices = img_data_full.shape[2]
        
        selected_slice_indices = []
        if num_original_slices == 0:
            # This case is already handled earlier, but as a safeguard
            print(f"Error: num_original_slices is 0 in selection logic for {nii_path}. Should have been caught.")
            return saved_slices_info

        if num_original_slices > 100:
            # Generate 100 evenly spaced indices from the original slices
            selected_slice_indices = np.linspace(0, num_original_slices - 1, 100, dtype=int)
        elif num_original_slices < 100:
            # Use all original slices and pad by repeating the last slice
            selected_slice_indices = list(range(num_original_slices))
            last_slice_idx = num_original_slices - 1
            padding_needed = 100 - num_original_slices
            selected_slice_indices.extend([last_slice_idx] * padding_needed)
        else: # num_original_slices == 100
            selected_slice_indices = list(range(100))

        for i in range(100): # This loop now runs 100 times
            original_slice_idx = selected_slice_indices[i]
            
            if img_data_full.ndim == 3:
                img_slice_raw = img_data_full[:, :, original_slice_idx]
            elif img_data_full.ndim == 4:
                img_slice_raw = img_data_full[:, :, original_slice_idx, 0]
            
            img_slice_oriented = np.rot90(img_slice_raw, k=1)

            if not is_mask: # This function is primarily called with is_mask=False for images
                min_val, max_val = np.min(img_slice_oriented), np.max(img_slice_oriented)
                if max_val - min_val > 1e-5:
                    img_slice_normalized = (img_slice_oriented - min_val) / (max_val - min_val)
                else:
                    img_slice_normalized = np.zeros_like(img_slice_oriented)
                img_slice_8bit = (img_slice_normalized * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_slice_8bit).convert('L')
            else:
                # This branch is kept for completeness but typically not hit for image conversion path
                pil_img = Image.fromarray(img_slice_oriented.astype(np.uint8)).convert('L')

            slice_filename = f"{base_filename}_slice_{i:04d}.jpg" # 'i' is the new slice index from 0 to 99
            jpg_filepath = os.path.join(output_dir, slice_filename)
            pil_img.save(jpg_filepath)
            saved_slices_info.append((jpg_filepath, pil_img.width, pil_img.height))
        
        print(f"[nii_to_jpg_slices] Finished slice processing loop.")
            
    except Exception as e:
        print(f"Error converting slices from {nii_path}: {e}")
    print(f"[nii_to_jpg_slices] Finished processing {nii_path}.")
    return saved_slices_info

def mask_slice_to_polygons(mask_slice_data):
    """
    Converts a single 2D mask slice (numpy array) to COCO polygon format.
    Each distinct polygon becomes a separate annotation.
    Returns a list of tuples, where each tuple is (segmentation_list_for_single_polygon, label_val, bbox, area).
    """
    annotations_data = [] # List of (segmentation_list, label_val, bbox, area)
    if mask_slice_data.ndim != 2:
        print(f"Error: mask_slice_to_polygons expects a 2D array, got {mask_slice_data.ndim}D")
        return annotations_data, 0, 0

    print(f"        [mask_slice_to_polygons] Start processing slice.")
    mask_slice_oriented = np.rot90(mask_slice_data, k=1)
    mask_slice_uint8 = mask_slice_oriented.astype(np.uint8)
    
    unique_labels = np.unique(mask_slice_uint8)
    print(f"        [mask_slice_to_polygons] Found unique labels: {unique_labels}.")

    for label_val in unique_labels:
        if label_val == 0:  # Skip background
            continue
        
        current_label_mask = (mask_slice_uint8 == label_val).astype(np.uint8)
        contours = measure.find_contours(current_label_mask, 0.5, positive_orientation='low')
        
        for contour in contours:
            # Ensure contour is valid for polygon creation
            if contour.shape[0] < 3: # A polygon needs at least 3 points
                continue

            # skimage gives (row, col), convert to (x, y) for shapely
            flipped_contour = np.fliplr(contour) 
            
            # Create polygon
            poly = Polygon(flipped_contour)
            
            # Simplify polygon
            simplified_poly = poly.simplify(1.0, preserve_topology=False)

            polys_to_process = []
            if simplified_poly.is_empty or not simplified_poly.is_valid:
                continue
            
            if simplified_poly.geom_type == 'Polygon':
                polys_to_process.append(simplified_poly)
            elif simplified_poly.geom_type == 'MultiPolygon':
                for p_geom in simplified_poly.geoms: # Access individual polygons within MultiPolygon
                    if p_geom.is_empty or not p_geom.is_valid or p_geom.geom_type != 'Polygon':
                        continue
                    polys_to_process.append(p_geom)
            
            # Process each individual valid polygon
            for individual_poly_obj in polys_to_process:
                segmentation_coords = np.array(individual_poly_obj.exterior.coords).ravel().tolist()

                s_np = np.array(segmentation_coords).reshape(-1,2)
                s_np = np.maximum(s_np, 0) 
                if s_np.shape[0] < 3:
                    continue
                
                final_segmentation_for_coco = [s_np.ravel().tolist()]

                x_min, y_min, x_max, y_max = individual_poly_obj.bounds
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min] 
                area = individual_poly_obj.area

                if area > 0: 
                    annotations_data.append((final_segmentation_for_coco, int(label_val), bbox, area))
            
    print(f"        [mask_slice_to_polygons] Finished processing slice. Found {len(annotations_data)} individual polygons.")
    return annotations_data, mask_slice_uint8.shape[1], mask_slice_uint8.shape[0]

def convert_nii_to_coco(nii_images_dir, nii_masks_dir, output_coco_dir, dataset_name="train", existing_coco_data=None):
    """
    Main function to convert NIfTI dataset to COCO format.
    Can append to existing COCO data if provided.
    """
    output_images_dir = os.path.join(output_coco_dir, "JPEGImages")
    output_annot_file = os.path.join(output_coco_dir, "annotations.json")

    os.makedirs(output_images_dir, exist_ok=True)

    if existing_coco_data:
        coco_output = existing_coco_data
        # Determine starting IDs from existing data
        if coco_output['images']:
            image_id_counter = max(img['id'] for img in coco_output['images'])
        else:
            image_id_counter = 0
        if coco_output['annotations']:
            annotation_id_counter = max(ann['id'] for ann in coco_output['annotations'])
        else:
            annotation_id_counter = 0
        print(f"Appending to existing COCO data. Starting image_id: {image_id_counter+1}, annotation_id: {annotation_id_counter+1}")
    else:
        coco_output = {
            "info": {
                "description": "Converted NIfTI dataset to COCO format",
                "url": "",
                "version": "1.0",
                "year": datetime.date.today().year,
                "contributor": "AI Assistant",
                "date_created": datetime.datetime.utcnow().isoformat(' ')
            },
            "licenses": [{
                "url": "",
                "id": 0,
                "name": ""
            }],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "liver", "supercategory": "organ"},
                {"id": 2, "name": "right-kidney", "supercategory": "organ"},
                {"id": 3, "name": "spleen", "supercategory": "organ"},
                {"id": 4, "name": "pancreas", "supercategory": "organ"},
                {"id": 5, "name": "aorta", "supercategory": "organ"},
                {"id": 6, "name": "ivc", "supercategory": "organ"},
                {"id": 7, "name": "rag", "supercategory": "organ"},
                {"id": 8, "name": "lag", "supercategory": "organ"},
                {"id": 9, "name": "gallbladder", "supercategory": "organ"},
                {"id": 10, "name": "esophagus", "supercategory": "organ"},
                {"id": 11, "name": "stomach", "supercategory": "organ"},
                {"id": 12, "name": "duodenum", "supercategory": "organ"},
                {"id": 13, "name": "left kidney", "supercategory": "organ"}
            ]
        }
        image_id_counter = 0
        annotation_id_counter = 0
        print(f"Creating new COCO data. Starting image_id: {image_id_counter+1}, annotation_id: {annotation_id_counter+1}")

    # Ensure nii_image_files are correctly filtered and sorted
    nii_image_files = sorted([f for f in os.listdir(nii_images_dir) if f.endswith(".nii.gz") and os.path.isfile(os.path.join(nii_images_dir, f))])
    
    # Variables to store info for the very first slice of the first image for debugging
    first_slice_jpg_path_for_debug = None
    first_slice_annotations_for_debug = None

    for idx, nii_img_filename_full in enumerate(tqdm(nii_image_files, desc=f"Processing NIfTI files for {dataset_name}")):
        print(f"\nProcessing file {idx+1}/{len(nii_image_files)}: {nii_img_filename_full}")
        nii_img_filepath = os.path.join(nii_images_dir, nii_img_filename_full)
        
        # Construct mask filename based on image filename
        # Assuming image is 'FLARE22_Tr_xxxx_0000.nii.gz' and mask is 'FLARE22_Tr_xxxx.nii.gz'
        if "_0000.nii.gz" in nii_img_filename_full:
            mask_filename_base = nii_img_filename_full.replace("_0000.nii.gz", "")
            nii_mask_filename = mask_filename_base + ".nii.gz"
        else: # Fallback or other naming conventions
            nii_mask_filename = nii_img_filename_full 
            print(f"Warning: Image filename {nii_img_filename_full} does not follow expected '_0000.nii.gz' pattern. Assuming mask has the same name.")

        nii_mask_filepath = os.path.join(nii_masks_dir, nii_mask_filename)
        base_filename = nii_img_filename_full.replace(".nii.gz", "") # Used for JPG filenames

        if not os.path.exists(nii_mask_filepath):
            print(f"Warning: Mask file {nii_mask_filepath} (derived from {nii_img_filename_full}) not found, skipping.")
            continue

        saved_image_slices_info = nii_to_jpg_slices(nii_img_filepath, output_images_dir, base_filename, is_mask=False)

        if not saved_image_slices_info: # If nii_to_jpg_slices returned empty (e.g. 0 slices in image NIfTI)
            print(f"Skipping {nii_img_filename_full} due to no image slices processed.")
            continue

        mask_data_full = None
        num_original_mask_slices = 0
        try:
            print(f"Attempting to load NIfTI mask: {nii_mask_filepath}")
            nii_mask_img = nib.load(nii_mask_filepath)
            print(f"Successfully loaded NIfTI mask: {nii_mask_filepath}")
            mask_data_full = nii_mask_img.get_fdata()
            if mask_data_full.ndim not in [3, 4]:
                print(f"Unsupported NIfTI mask dimension: {mask_data_full.ndim} for {nii_mask_filepath}. Using blank masks for this file.")
                mask_data_full = None # Signal to use blank masks
            if mask_data_full is not None:
                 num_original_mask_slices = mask_data_full.shape[2]
                 if num_original_mask_slices == 0:
                    print(f"Mask file {nii_mask_filepath} has 0 slices. Using blank masks.")
                    mask_data_full = None # Treat as if loading failed for 0-slice masks
        except Exception as e:
            print(f"Error loading mask {nii_mask_filepath}: {e}. Using blank masks for this file.")
            mask_data_full = None
            num_original_mask_slices = 0 # Redundant but clear

        for coco_slice_idx, (jpg_filepath, width, height) in enumerate(saved_image_slices_info): # coco_slice_idx from 0 to 99
            image_id_counter += 1
            coco_output["images"].append({
                "id": image_id_counter,
                "file_name": os.path.basename(jpg_filepath),
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.datetime.utcnow().isoformat(' ')
            })

            actual_mask_slice_data_unrotated = None
            if mask_data_full is None or num_original_mask_slices == 0:
                actual_mask_slice_data_unrotated = np.zeros((height, width), dtype=np.uint8)
            else:
                # Determine the corresponding original mask slice index based on coco_slice_idx (0-99)
                # This logic should mirror the one in nii_to_jpg_slices for selecting image slices
                mask_selected_indices = []
                if num_original_mask_slices > 100:
                    mask_selected_indices = np.linspace(0, num_original_mask_slices - 1, 100, dtype=int)
                elif num_original_mask_slices < 100:
                    mask_selected_indices = list(range(num_original_mask_slices))
                    last_mask_idx = num_original_mask_slices - 1
                    padding_needed = 100 - num_original_mask_slices
                    mask_selected_indices.extend([last_mask_idx] * padding_needed)
                else: # num_original_mask_slices == 100
                    mask_selected_indices = list(range(100))
                
                current_mask_original_idx = mask_selected_indices[coco_slice_idx]

                if mask_data_full.ndim == 3:
                    actual_mask_slice_data_unrotated = mask_data_full[:, :, current_mask_original_idx]
                elif mask_data_full.ndim == 4:
                    actual_mask_slice_data_unrotated = mask_data_full[:, :, current_mask_original_idx, 0]
            
            if actual_mask_slice_data_unrotated is None:
                 # This case should ideally not be reached if logic above is correct
                 print(f"Critical error: actual_mask_slice_data_unrotated is None for {jpg_filepath}. Using blank mask.")
                 actual_mask_slice_data_unrotated = np.zeros((height, width), dtype=np.uint8)

            slice_annotations_data, _, _ = mask_slice_to_polygons(actual_mask_slice_data_unrotated)

            for segmentation_list, label_val, bbox, area in slice_annotations_data:
                annotation_id_counter += 1
                coco_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": int(label_val),
                    "segmentation": segmentation_list,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })

            # Store info for the very first slice of the first image for later debugging
            if first_slice_jpg_path_for_debug is None and coco_slice_idx == 0:
                first_slice_jpg_path_for_debug = jpg_filepath
                first_slice_annotations_for_debug = slice_annotations_data

    # After processing all files, create the debug overlay for the first slice of the first image
    if first_slice_jpg_path_for_debug and first_slice_annotations_for_debug:
        try:
            img_for_debug = cv2.imread(first_slice_jpg_path_for_debug)
            if img_for_debug is None:
                print(f"Debug: Failed to load image {first_slice_jpg_path_for_debug} for overlay.")
            else:
                if first_slice_annotations_for_debug:
                    for single_polygon_annotation in first_slice_annotations_for_debug:
                        # single_polygon_annotation is (segmentation_list_for_single_polygon, label_val, bbox, area)
                        # segmentation_list_for_single_polygon is [[x1,y1,x2,y2,...]]
                        coco_segmentation = single_polygon_annotation[0] 
                        if coco_segmentation and coco_segmentation[0]: # Ensure it's not empty
                            poly_coords_flat = coco_segmentation[0] # Get the flat list [x1,y1,...]
                            contour = np.array(poly_coords_flat).reshape((-1, 1, 2)).astype(np.int32)
                            cv2.polylines(img_for_debug, [contour], isClosed=True, color=(0, 255, 0), thickness=1)
                
                debug_img_path = os.path.join(os.path.dirname(output_coco_dir), "debug_overlay_first_slice.jpg")
                cv2.imwrite(debug_img_path, img_for_debug)
                print(f"Debug: Saved overlay image of the first slice of the first image to {debug_img_path}")
        except Exception as e_debug:
            print(f"Debug: Error creating overlay image for the first slice: {e_debug}")

    with open(output_annot_file, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"COCO dataset generated at {output_coco_dir}")
    print(f"Total images: {len(coco_output['images'])}")
    print(f"Total annotations: {len(coco_output['annotations'])}")

if __name__ == '__main__':
    # --- Configuration for original dataset ---
    ORIGINAL_INPUT_NII_IMAGES_DIR = "/mnt/gemlab_data_3/User_database/liangzhichao/task2/train_gt_label/imagesTr/"
    ORIGINAL_INPUT_NII_MASKS_DIR = "/mnt/gemlab_data_3/User_database/liangzhichao/task2/train_gt_label/labelsTr/"
    OUTPUT_COCO_DIR = "/home/data/liangzhichao/Code/organseg/data/task/train"
    ORIGINAL_DATASET_SPLIT_NAME = "train_original"

    # --- Configuration for pseudo-label dataset ---
    PSEUDO_INPUT_NII_IMAGES_DIR = "/mnt/gemlab_data_3/User_database/liangzhichao/task2/train_pseudo_label/imagesTr/"
    PSEUDO_INPUT_NII_MASKS_DIR = "/mnt/gemlab_data_3/User_database/liangzhichao/task2/train_pseudo_label/labelsTr/"
    # Output COCO directory is the same, as we are appending
    PSEUDO_DATASET_SPLIT_NAME = "train_pseudo"

    # Ensure output directory exists
    os.makedirs(OUTPUT_COCO_DIR, exist_ok=True)
    annotations_file_path = os.path.join(OUTPUT_COCO_DIR, "annotations.json")

    # --- Optional: Process original dataset first (if you want to regenerate or run it first) ---
    # Comment out this block if you only want to append pseudo-label data to an existing annotations.json
    # or if the original data is already processed.
    # print(f"Starting conversion for original dataset...")
    # print(f"Input NIfTI Images: {ORIGINAL_INPUT_NII_IMAGES_DIR}")
    # print(f"Input NIfTI Masks: {ORIGINAL_INPUT_NII_MASKS_DIR}")
    # print(f"Output COCO Data: {OUTPUT_COCO_DIR}")
    # # For the first run, existing_coco_data is None or the file doesn't exist
    # existing_data_for_original = None
    # if os.path.exists(annotations_file_path):
    #     # If you intend to overwrite or start fresh for original, delete annotations_file_path or handle accordingly
    #     # For now, let's assume if it exists, we might be re-running, so we start fresh for original part.
    #     # Or, more safely, if you always want original to be fresh, ensure annotations_file_path is removed before this call.
    #     # To simply run original data processing and create a new annotations.json:
    #     print(f"Processing original dataset. This will create/overwrite {annotations_file_path} if it's the first dataset being processed.")
    # else:
    #     print(f"Processing original dataset. {annotations_file_path} does not exist, will be created.")
    # convert_nii_to_coco(ORIGINAL_INPUT_NII_IMAGES_DIR, ORIGINAL_INPUT_NII_MASKS_DIR, OUTPUT_COCO_DIR, ORIGINAL_DATASET_SPLIT_NAME, existing_coco_data=None)
    # print("Original dataset processing finished.")
    # print("-------------------------------------")

    # --- Process pseudo-label dataset, appending to existing annotations ---
    print(f"Starting conversion for pseudo-label dataset...")
    print(f"Input NIfTI Images: {PSEUDO_INPUT_NII_IMAGES_DIR}")
    print(f"Input NIfTI Masks: {PSEUDO_INPUT_NII_MASKS_DIR}")
    print(f"Output COCO Data: {OUTPUT_COCO_DIR} (appending)")

    existing_coco_content = None
    if os.path.exists(annotations_file_path):
        try:
            with open(annotations_file_path, 'r') as f:
                existing_coco_content = json.load(f)
            print(f"Successfully loaded existing annotations from {annotations_file_path} for appending.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {annotations_file_path}. Starting with fresh annotations for pseudo-labels, potentially overwriting.")
            existing_coco_content = None # Treat as if file didn't exist or was invalid
        except Exception as e:
            print(f"Error loading {annotations_file_path}: {e}. Starting with fresh annotations for pseudo-labels.")
            existing_coco_content = None
    else:
        print(f"{annotations_file_path} does not exist. Pseudo-label data will form the basis of a new annotations file if original data was not processed first.")

    convert_nii_to_coco(PSEUDO_INPUT_NII_IMAGES_DIR, PSEUDO_INPUT_NII_MASKS_DIR, OUTPUT_COCO_DIR, PSEUDO_DATASET_SPLIT_NAME, existing_coco_data=existing_coco_content)

    print("Pseudo-label dataset processing finished.")
    print("Script finished.")

# ... existing code ...