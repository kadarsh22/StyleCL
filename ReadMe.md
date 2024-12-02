## StyleCL Setup
1) Create new conda environment named 'StyleCL'
2) Install the necessary requirements for Stylegan2-ADA as mentioned by the authors of Stylegan2-ADA pytorch implementation.
3) Install pytorch (version tested by us) : pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
4) Alternatively you can also use the requirements.txt file for installing the necessary dependencies

## StyleCL Usage Instructions
To learn the StyleCL with dictionary learning alone, use the code provided in the 'stylecl_only_dict_learning' folder and to learn the feature adaptor, use the code provided in the 'stylecl_feature_adaptor_learning' folder.

Follow the steps below to prepare your data:

1 .Create a folder named 'data' in the repository.

2 .Download the data and convert it to a zip file using the dataset_tool.py file. Place the converted data inside the 'data' folder.

To run StyleCL on the flowers dataset, use the following command:


CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --data data/flowers/flowers.zip --outdir results/flowers/ours/ --cfg auto --resume celebahq256 --metrics=fid50k_full > nohup_logs/flowers/ours.out &

Note that It takes approximately 3 hours to complete 50ticks of training, when running on Single GPU (RTX 3090)

## Lifelong Classification
1. Obtain the Generators for all the task by running StyleCL on these tasks. Place the trained models in the folder "pretrained_models/lifelong_classification/$task_name"
2. change do_method variable at line 103 to the required method (eg. "StyleCL")
4. Run lifelong_classification.py file to _quant_results.mat file for each particular algorithm
5. Run plot_accuracy.py to get the plot