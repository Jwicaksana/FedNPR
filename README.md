# FedNPR
<br>
Code for FedNPR (non-parametric regularization for federated medical image classification)
<br>
Code structure is as follow:
---

```bash
 |--> FedNPR
 |--> datasets
   |-->ich
     |--> images
   |-->isic
     |-->isic2019_preprocessed
```

Regarding data preprocessing:
1) ICH: data is split with the hyperparameters listed in the manuscript. Images are preprocessed and placed in images folder. 
2) ISIC: follows <a href="https://github.com/owkin/FLamby">FLAmby</a> on Fed-ISIC2019, including the train-test split ([isic]train_test_split). Images are placed in isic2019_preprocessed.

