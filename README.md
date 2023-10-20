# Contrastive Learning Verification

## Introduction
This is a PyTorch implementation for Contrastive Learning Verification.  

![image](https://github.com/byunghyun23/contrastive-learning/blob/main/assets/fig1.png)

We used CIFAR-10 dataset.  

![image](https://github.com/byunghyun23/contrastive-learning/blob/main/assets/fig2.png)

## Architecture
![image](https://github.com/byunghyun23/contrastive-learning/blob/main/assets/fig3.PNG)

## Run
```
python run.py
```
After training, CLR(Contrastive Learning Representations) model and FT(Fine-tuning) model are generated.

## Results
![image](https://github.com/byunghyun23/contrastive-learning/blob/main/assets/fig4.png)
| Type | Epochs | Feature dimensionality  | Projection Head dimensionality | Valid Accuracy (Best)          |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Contrastive Learning (FT.pth)          | 30          | 512         | 256         | 0.7352         |

