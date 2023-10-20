# Contrastive Learning

## Introduction
This is a PyTorch implementation for Contrastive Learning.  

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
| Without Contrastive Learning          | 50          | 512         | 256         | 0.7000         |
| With Contrastive Learning          | 50          | 512         | 256         | 0.7269         |

