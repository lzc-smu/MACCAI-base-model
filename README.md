# CylinGCN: Cylindrical Structures Segmentation in 3D Biomedical Optical Imaging by a contour-based Graph Convolutional Network

![city](assets/SFEGCN.jpg)

> [**Organ-level instance segmentation enables continuous time-space-spectrum analysis of photoacoustic tomography images**](https://www.sciencedirect.com/science/article/abs/pii/S136184152400327X/)  
> Zhichao Liang, Shuangyang Zhang, Zongxin Mo, Xiaoming Zhang, Anqi Wei, Wufan Chen, Li Qi

This repository is the official implementation of "CylinGCN: Cylindrical Structures Segmentation in 3D Biomedical Optical Imaging by a contour-based Graph Convolutional Network".

Any questions or discussions are welcomed!

## Environments and Requirements

*   **Operating System**: [Please specify, e.g., Ubuntu 20.04, Windows 10]
*   **CPU**: [Please specify, e.g., Intel Core i7]
*   **RAM**: [Please specify, e.g., 16GB]
*   **GPU**: [Please specify, e.g., NVIDIA GeForce RTX 3090]
*   **CUDA Version**: [Please specify, e.g., 11.x]
*   **Python Version**: [Please specify, e.g., 3.7, 3.8]

To install requirements:

```bash
pip install -r requirements.txt
```

For detailed installation instructions, please refer to [INSTALL.md](assets/INSTALL.md).

## Dataset

*   **Data Source**: [Please provide a link to download the data if publicly available, or describe the source]
*   **Data Preparation**: [Please describe how to prepare the data, e.g., folder structures, any conversion scripts like `convert_nii_to_coco.py` if used for initial data setup]
    *   Example structure:
        ```
        data/
        ├── space/
        │   ├── test/
        │   └── train/
        └── task/
            ├── test/
            └── train/
        ```

## Preprocessing

[Please provide a brief description of the preprocessing methods used, e.g., cropping, intensity normalization, resampling. If you have a dedicated preprocessing script, mention it here with example usage.]

Example of running a preprocessing script (if applicable):

```bash
# python preprocessing_script.py --input_path <path_to_raw_data> --output_path <path_to_preprocessed_data>
```

## Training

To train the model(s) in the paper, run this command:

```bash
python train.py --task space --input_mode space --dataset pat --arch dlagcnmulti_34 [add_other_relevant_hyperparameters_here]
```

*   Please specify the full training procedure and all relevant hyper-parameters used to achieve the reported results.
*   You can download trained models here: [Provide a link to download your trained model, e.g., Google Drive, Zenodo]

## Inference (Testing)

To run inference on test data, use the following command:

```bash
python test.py --demo <path_to_test_data_or_JPEGImages> --load_model <path_to_your_trained_model e.g., exp/checkpoints/task_arch/model_best.pth> --arch dlagcnmulti_34 --dataset pat --output_imgs [add_other_relevant_options_here]
```

*   `<path_to_test_data_or_JPEGImages>`: Specify the path to your test dataset or a directory of images.
*   `<path_to_your_trained_model>`: Specify the path to the pre-trained model weights.
*   The results (e.g., output images, segmentation masks) will be saved in a directory (e.g., `exp/results/imgs/` by default, or as specified by options).

## Evaluation

[Please describe how to evaluate the inference results to obtain the metrics reported in your paper. If you have an evaluation script, provide the command to run it.]

Example of running an evaluation script (if applicable):

```bash
# python eval.py --seg_data <path_to_inference_results> --gt_data <path_to_ground_truth>
```

## Results

[Please include a table or a summary of the main results from your paper. You can link to a leaderboard if applicable, or include key figures.]

Example Table:

| Metric       | Value  |
| ------------ | :----: |
| DICE Score   |  XX.X% |
| IoU          |  YY.Y% |
| ...          |  ...   |

## Acknowledgement

Our work benefits a lot from [CenterTrack](https://github.com/xingyizhou/CenterTrack#tracking-objects-as-points) and [DeepSnake](https://github.com/zju3dv/snake). Thanks for their great contributions.

We also thank the contributors of any public datasets used in this research.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{Liang2024,
  title={Organ-level instance segmentation enables continuous time-space-spectrum analysis of pre-clinical abdominal photoacoustic tomography images},
  author={Liang, Zhichao and Zhang, Shuangyang and Mo, Zongxin and Zhang, Xiaoming and Wei, Anqi and Chen, Wufan and Qi, Li},
  journal={Medical Image Analysis},
  volume={101},
  pages={103402},
  year={2025},
  publisher={Elsevier} 
}
```
