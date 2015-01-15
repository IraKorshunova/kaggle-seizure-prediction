#Seizure prediction using convolutional neural networks

##Introduction

This document discusses my approach for the [Kaggle Seizure Prediction Challenge](http://www.kaggle.com/c/seizure-prediction), which resulted in the 10th place in the ranking. The goal of the competition was to classify 10 minute intracranial EEG (iEEG) data clips into *preictal* for pre-seizure data or *interictal* for non-seizure data segments. For this problem I used convolutional neural networks (convnets). The following description provides some excerpts from my master's thesis, which is available [here](kaggle-seizure-prediction/thesis.pdf).

##Features
TODO

##Network architecture
TODO


![Figure 1](/images/model2_annot.png)


##Model averaging
TODO
[link](https://github.com/IraKorshunova/kaggle-seizure-prediction/tree/master/settings_dir) 


##Code
It's a beautiful Python+Theano code, however not optimized to run on GPU. Its description I will add later if someone is interested.

## References
1. Howbert JJ, Patterson EE, Stead SM, Brinkmann B, Vasoli V, Crepeau D, Vite CH, Sturges B, Ruedebusch V, Mavoori J, Leyde K, Sheffield WD, Litt B, Worrell GA (2014) Forecasting seizures in dogs with naturally occurring epilepsy. PLoS One 9(1):e81920.

