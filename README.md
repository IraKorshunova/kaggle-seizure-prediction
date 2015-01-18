#Seizure prediction using convolutional neural networks

##Introduction

This document roughly discusses my approach for the [Kaggle Seizure Prediction Challenge](http://www.kaggle.com/c/seizure-prediction), which resulted in the 10th place in the ranking. The goal of the competition was to classify 10 minute intracranial EEG (iEEG) data clips into *preictal* for pre-seizure data or *interictal* for non-seizure data segments. For this problem I used convolutional neural networks (convnets). The following description provides a few excerpts from my master's thesis, which is available [here](kaggle-seizure-prediction/thesis.pdf).

##Features
The first preprocessing steps were to resample the signal to 400 Hz and apply a band-pass filter between 0.1-180 Hz.

Previous studies showed that features from the frequency domain are effective for seizure prediction (Howbert et al., 2014), so we followed the similar ideas. Each 10 minute time series was partitioned into nonoverlapping 1 min frames, and within each frame we calculated log_{10} of its amplitude spectrum. The spectrum was divided into 6 frequency bands: delta (0.1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz), low­gamma (30-70 Hz) and high­gamma (70-180 Hz). In each frequency band we took an average of the corresponding amplitudes. Thus, the dimension of the data clip is equal to Nx6x10 (channels x  frequency bands x time frames). 
Additionally, in some of our models we used standard deviation of a signal, calculated within the same time windows as FFT. Several models were also trained on FFT data partitioned into 8 bands by splitting wide gamma bands into 2 parts: low-gamma to 30-50Hz and 50-70Hz, high-gamma to 70-100Hz and 100-180Hz. 

##Network architecture
In the way data was collected and due to nonstationarity of EEG signal, signs of preictal activity may not be present during the whole 10 minute clip. Therefore, we want our model to learn features localized in time and then combine the information from different time slices. This idea was implemented with a convolutional neural network, which performs one-dimensional convolution through time, thus extracting the same types of features from each time frame separately, and then combines the information across the time axis in higher layers. 

Moreover, it is important to consider the relationships between pairs of EEG channels (Mirowski et al., 2009). One option is to extend a feature set with pairwise features e.g. cross-correlations. However, our idea was to let the filters in the first convolutional layer to see all frequency bands of all the channels at a particular time frame, so they could learn relevant relationships by itself.

The basic architecture we used is shown below.
![Figure 1](/images/model2_annot.png)

Its first layer (C1) performs convolution in a time dimension over all N channels and all 6 frequency bands, so the shape of its filters is 6xNx1. C1 has 16 feature maps each of shape 1x10. The second layer (C2) performs convolution with filters 16x2, so the width of the resulting feature map is 10-2+1=9. Second layer has 32 feature maps. C2 is followed by a global temporal pooling layer GP3, which computes the following statistics: mean, maximum, minimum, variance, geometrical mean, L2 norm over 9 values in each feature map from C2. GP3 is fully connected with 128 units in F4 layer; output of the network is a logistic regression unit. 

 Rectified linear units were used in C1 and C2 layers and tanh activation in the hidden layer.

## Training
We used  mini-batches of 10 examples and trained the network with ADADELTA method. Inputs to the network were previously standardized. Dropout was used in last 2 layers.

##Model averaging
As a final model we used geometric average of predictions (normalized to 0-1 range for each subject) obtained from 11 convnets. See 
[setting_dir](https://github.com/IraKorshunova/kaggle-seizure-prediction/tree/master/settings_dir) for exact parameters of each convnet.


##Code
It's a beautiful Python+Theano code, however not optimized to run on GPU.

## References
1. Howbert JJ, Patterson EE, Stead SM, Brinkmann B, Vasoli V, Crepeau D, Vite CH, Sturges B, Ruedebusch V, Mavoori J, Leyde K, Sheffield WD, Litt B, Worrell GA (2014) Forecasting seizures in dogs with naturally occurring epilepsy. PLoS One 9(1):e81920.
2. Mirowski, P., D. Madhavan, Y. LeCun, and R. Kuzniecky (2009). Classification of patterns of EEG synchronization for seizure prediction. Clinical neurophysiology 120(11), 1927–1940.
