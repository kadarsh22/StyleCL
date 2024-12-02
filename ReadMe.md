
# StyleCL: Setup and Usage Guide

## Table of Contents
1. [Setup](#setup)
2. [Usage Instructions](#usage-instructions)
   - [Data Preparation](#data-preparation)
   - [Running StyleCL](#running-stylecl)
3. [Lifelong Classification](#lifelong-classification)

---

## Setup

1. **Create a Conda Environment**:  
   Set up a new Conda environment named `StyleCL`:
   ```bash
   conda create -n StyleCL python=3.x  # Replace `3.x` with the desired Python version
   conda activate StyleCL
   ```

2. **Install StyleGAN2-ADA Requirements**:  
   Follow the [official instructions](https://github.com/NVlabs/stylegan2-ada-pytorch) for setting up the StyleGAN2-ADA PyTorch implementation.

3. **Install PyTorch**:  
   Use the tested version of PyTorch with CUDA 11.1 support:
   ```bash
   pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. **Install Dependencies via Requirements File (Optional)**:  
   Alternatively, install all necessary dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

### Data Preparation

1. **Create Data Directory**:  
   Inside the repository, create a folder named `data`:
   ```bash
   mkdir data
   ```

2. **Prepare the Dataset**:  
   Download your dataset, convert it into a compatible zip format using `dataset_tool.py`, and place the zip file inside the `data` directory:
   ```bash
   python dataset_tool.py --source <source_folder> --dest data/<dataset_name>.zip
   ```

### Running StyleCL

- **Dictionary Learning Only**:  
  Use the code in the `stylecl_only_dict_learning` folder.

- **Feature Adaptor Learning**:  
  Use the code in the `stylecl_feature_adaptor_learning` folder.

#### Example: Training on the Flowers Dataset
Run the following command to train StyleCL on the Flowers dataset:
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py \
    --data data/flowers/flowers.zip \
    --outdir results/flowers/ours/ \
    --cfg auto \
    --resume celebahq256 \
    --metrics fid50k_full > nohup_logs/flowers/ours.out &
```

- **Training Duration**: Approximately 3 hours for 50 ticks on a single GPU (RTX 3090).

---

## Lifelong Classification

1. **Prepare Trained Generators**:  
   Obtain trained generators for all tasks by running StyleCL on those tasks. Place the resulting models in:
   ```
   pretrained_models/lifelong_classification/$task_name
   ```

2. **Set Classification Method**:  
   Modify the `do_method` variable (line 103 in the script) to the desired method, e.g., `"StyleCL"`.

3. **Run Lifelong Classification**:  
   Execute the `lifelong_classification.py` script to generate the `_quant_results.mat` file for each algorithm:
   ```bash
   python lifelong_classification.py
   ```

4. **Plot Accuracy Results**:  
   Generate accuracy plots using:
   ```bash
   python plot_accuracy.py
   ```

---

Feel free to reach out for further assistance or clarification on any step. Happy experimenting with StyleCL! 🚀