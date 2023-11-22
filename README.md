
# AERMNet: Attention-enhanced relational memory network for medical image report generation

## Environment Requirements:
---
* `python==3.7`<br>  
* `pytorch==1.1.0`<br>
* `torchvision==0.3.0`<br>

## Data preparation
---
To run the code, you can download IU X-Ray and MIMIC-CXR datasets.<br>

* For IU X-Ray, you can download the dataset from [IU X-Ray link](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing"悬停显示") and then put the data files in yourdata/iu_xray.<br>

* For MIMIC-CXR, you can download the dataset from [MIMIC-CXR link](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing"悬停显示") and then put the data files in yourdata/mimic_cxr. You can apply the dataset here with your license of PhysioNet.<br>

NOTE: The IU X-Ray dataset is of small size, and thus the variance of the results is large.<br> 

## Training procedure
---
Run python train_AERMNet.py using the following arguments:<br>
|Argument|Possible values|
|:----:|:----:|
|--epochs |the number of epochs (default: 50) |
|--batch_size|Batch size (default: 32) |
|--workers |Number of workers (default: 2)|
|--data_folder|Folder path to save files|
|--data_folder_new|Folder path where checkpoint files are saved|
|--lr|learning rate|
|--decoder_dim|Number of decoder layers|
|--alpha_c|Regularization parameters|
Run `train_AERMNet.py` to train a model on the IU X-Ray data and the MIMIC-CXR data <br> 

## Test
Run `test_AERMNet.py` to test AERMNet model on the IU X-Ray data and the MIMIC-CXR data. <br> 


