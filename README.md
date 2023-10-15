# SwinIR
SwinIR: Image Restoration Using Swin Transformer

- **data** : contains python file for each type of dataset for example `dataset_sr` is for super-resolution dataset
- **models** : this folder contains models files and core of this repo
    - for adding network to this repo you have to do these steps:
        - create your network like swinir
        - add your network to the `select_network.py`
    - for adding new loss or other changes that related to the training like feeding network or etc you will find your answer in `model_plain.py`
- **options**: these folder contains the training configs
- `main_train_psnr.py`: this file is for training and for run this file:
    - **args**
        - `--opt` for choosing option file
        - `--tensorboard` for using tensorboard for visualisation
- `main_test_swinir.py` for testing your model
    -   **args**
        - `--scale`
        - `--training_patch_size`
        - `--model_path` : path to your model
        - `--folder_lq`: path to low quality folder
        - `--folder_gt`: path to ground truth folder
- **scripts**: scripts for preparing your data for train or test
    - **train** : 2 steps
        - run `generate_mod_LR_bic.py` on your original image of your dataset for example div2k
        - run `extract_subimages.py` on HR and LR folders for extracting subimages
    - **test** : 2 steps
        - run `generate_mod_LR_bic.py` on your original image of your dataset for example set5
    - **downloading benchmarks**: for downloading benchmarks you have to run this command 
        - `python download_benchmarks.py zip_url https://drive.google.com/file/d/1Nri53Tq9XcAX2S9MdXCC3GyvTHHVPnD9/view\?usp\=drive_link destination_folder ../../testsets/`
        