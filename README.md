#Seizure prediction using convolutional neural networks

##Introduction

This document discusses my approach for the [Kaggle Seizure Prediction Challenge](http://www.kaggle.com/c/seizure-prediction), which resulted in the 10th place in the ranking. The goal of the competition was to classify 10 minute intracranial EEG (iEEG) data clips into *preictal* for pre-seizure data or *interictal* for non-seizure data segments. For this problem I used convolutional neural networks (convnets). The following descriptions provides some excerpts from my master's thesis, which is available [here](kaggle-seizure-prediction/thesis.pdf).

##Features


##Network architecture

The architecture of the model, which appeared to be the best on a public leaderboard is shown on the Figure 1. It receives an input after the 1st normalization scheme, and its first layer performs convolution in a time dimension over all channels and all frequency bins, so the shape of its filters is *(64x1)*. Second convolutional layer is fully-connected with a hidden layer. Rectified linear units are used in all layers, and last 2 layers have a dropout of 0.2 and 0.5. On public/private leaderboard this model scored 0.81448/0.76256.

![Figure 1](/images/fig_1.png)

Intuitively the location of some feature in time should not be relevant, therefore I tried global temporal pooling with mean, max, min, variance, geometric mean and L2 norm calculated across time axis within each feature map in the 2nd convolutional layer. With global pooling layer previous model would look like on Figure 2. Experiments showed models with global pooling worked best with the 2nd normalization scheme, tanh activation in the hidden layer, stride of 2 in the 2nd layer and when using standard deviation of the signal as an additional feature.

![Figure 2](/images/fig_2.png)


Public LB scores were misleading by scoring no-global-pooling models higher than models with global pooling layer. It appeared to be completely opposite on the private leaderboard. However, public LB was the only source of validation as I used stratified split without keeping the sequences intact, thus getting very optimistic results. I tried to keep the sequences, but for some train-validation splits models were not training long enough (when using early-stopping). Later I started to train my models for a fixed number of updates and using data augmentation by overlapping clips from the same sequence on some number of time frames, e.g. if clip consists of 10 time frames, overlap of 9 frames yields approximately a 9 times bigger dataset.
To calibrate the predictions between subjects, I used min-max scaling on test probabilities, which used to improve the score on ~0.015 (although I think it was a bad idea to allow test data usage). 

##Model averaging

The submission, which finished in the 10th place and public/private score of 0.81292/0.78513 was an average prediction of the models with a global pooling layer and different combinations of parameters: number of kernels in convolutional layers, number of hidden units, amount of dropout, normalization scheme, number of time frames, overlap between clips, use of test data during normalizing the training data etc. Some of the settings files can be found [here](https://github.com/IraKorshunova/kaggle-seizure-prediction/tree/master/settings_dir) and list of ensembles is [here](https://github.com/IraKorshunova/kaggle-seizure-prediction/blob/master/utils/averager.py). In fact, I did a lot more experiments, and it took me a while to figure out a reasonable convnet architecture, but this discussion I will leave for my future thesis.


##Code description
It's a beautiful Python+Theano code, however not optimized to run on GPU. Its description I will add later if someone is interested.

## References
1. Howbert JJ, Patterson EE, Stead SM, Brinkmann B, Vasoli V, Crepeau D, Vite CH, Sturges B, Ruedebusch V, Mavoori J, Leyde K, Sheffield WD, Litt B, Worrell GA (2014) Forecasting seizures in dogs with naturally occurring epilepsy. PLoS One 9(1):e81920.

