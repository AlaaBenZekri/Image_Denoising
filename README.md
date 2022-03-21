# Image Denoising

## Introduction

The context of this notebook is an academic project where we were demanded to denoise images (without specifying the method). In the beginning I tried a non-ML solution that is called BM3D (Block Matching and 3D filtering), then I went ahead to experiment with the Deep Learning approach using the Auto-Encoder architecture.

## Denoising using BM3D

Block-matching and 3D filtering (BM3D) is considered a state-of-the-art approach among the non-local means filtering methodology that was first introduced in 2007 by Dabov et al [1].<br>
The algorithm is composed of two stages; in the first stage we divide the picture into patches, then each patch would be taken as a reference to build a 3 dimensional (or 4 dimensional in case of colored images) block grouping the reference with the most similar and not necessarily disjoint patches, afterwards the obtained block would go through the steps of 3D (or 4D) Fourrier Transform, hardthreshold filtering, inverse 3D (or 4D) Fourier Transform, block-wise estimation and in the end aggregation to replace the original patch. The initial output passes through a second stage with the only difference being the application of a Wiener filtering instead of the hardthresholding.<br>
![](https://raw.githubusercontent.com/AlaaBenZekri/Image_Denoising/main/bm3d.png)<br>
The implementation that we used for comparison with our work is based on the Makinen et al. 2019 paper [2] where significant ¨ improvements in terms of computation (and no noticeable reduction in performance) were introduced.<br>

## Denoising using Auto-Encoders

The particularity of this architecture is that it maps the input into a latent space finding an encoded representation, which is then decoded to get an output which is a reconstruction of the input. In other terms, given an input $X \in \mathbb{R}^{d}$, the encoding and decoding functions will be as follows :<br>
![](https://raw.githubusercontent.com/AlaaBenZekri/Image_Denoising/main/equations/eq1.JPG)<br>

and thus the model will try learn to minimize the difference between the original input X and its reconstruction X'.<br>

In our case, we will apply this architecture for the task of Image Denoising. The difference is that the provided input will be a corrupted version of the data that will be mapped into the clean data.<br>

The corruption can be modelled in various forms ; an additive Gaussian noise, a masking noise (a fraction of the input is randomly set to zero), salt-and-pepper noise (a fraction of the input is randomly set to its minimum or maximum value following a uniform distribution)...<br>
For the sake of simplicity, let's consider the corruption as a white additive noise z[3]. Thus, the optimization problem in this case will be :<br>
![](https://raw.githubusercontent.com/AlaaBenZekri/Image_Denoising/main/equations/eq2.JPG)<br>
The model that we’ve developed in our project is based on an Auto-Encoder architecture with an encoder and a decoder that are composed of convolutional layers instead of fully connected ones :<br>
![](https://raw.githubusercontent.com/AlaaBenZekri/Image_Denoising/main/autoencodercnn.JPG)<br>

## Results and Interpretations

irst of all, let us introduce a metric that we will use to understand how accurately our models are reconstructing the corrupted images ; Peak Signal-to-Noise Ratio (PSNR).It's expression is defined as follows :<br>
![](https://raw.githubusercontent.com/AlaaBenZekri/Image_Denoising/main/equations/eq3.JPG)<br>
Where MAX_I denotes the maximum possible value of the image (1 in our case), and MSE represents the Mean Squared Error between two images.<br>
![](https://raw.githubusercontent.com/AlaaBenZekri/Image_Denoising/main/result.png)<br>
We can see that all of the used methods were successful in reconstructing the original image with a significant improvement of at least 11dB in terms of PSNR in comparison with the noisy image.<br>

Nevertheless, we can notice that the deep learning algorithms resulted in better reconstructions than the one produced using BM3D, not only in terms of improvement of PSNR (around 3dB) but also perceptually as the BM3D image was over-smoothed.<br>

Moreover, it seems that both Auto-Encoders resulted in almost identical results. However, we believe that we should delay our judgement until looking at further results.<br>
![](https://raw.githubusercontent.com/AlaaBenZekri/Image_Denoising/main/plot.png)<br>
From what we observe through the plot, the MSE Auto-Encoder performs better for lower levels of noise. Meanwhile, the MAE Auto-Encoder takes the lead, ever so slightly, when faced with higher levels of corruption.<br>

In addition, with no surprises, the models' performances decreased drastically with the increase of the noise (but even when the image was completely corrupted, the models managed to provide reconstructions with a PSNR similar to that of a sigma^2=0.2 corrupted image).<br>

Finally, we can see that the PSNR at \sigma^2=0.2 is higher for lower variance values. On one hand, we can understand that the model didn't generalize well with lower-values of variance than what it encountered in training. But on the other hand, it's reasonable due to the fact that the model only trained on that level of variance.<br>

## Repository Files

You can find in this repository Python scripts for :
- importing the data and preparing it : data.py
- the model class : model.py
- models' weights after training : `normal_ae_mse_30epochs.tar` and `normal_ae_mae_30epochs.tar`
- scripts for model's training and evaluation : `train_and_eval.py`
- needed functions : `utils.py`

## References

[1] K. D. et al. (2008) Image denoising by sparse 3-d transform-domain collaborative filtering. [Online]. Available: https://www.researchgate.net/publication/6151802 Image_Denoising_by_Sparse_3-D_Transform-Domain_Collaborative_Filtering<br>
[2] Y. M. et al. (2019) Exact transform-domain noise variance for collaborative filtering of stationary correlated noise. [Online]. Available: https://webpages.tuni.fi/foi/papers/ICIP2019_Ymir.pdf<br>
[3] H. B. Y. M. P.-A. Vincent, Pascal; Larochelle, “Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion,” The Journal of Machine Learning Research, vol. 11, 2010. [Online]. Available: https://dl.acm.org/doi/pdf/10.5555/1756006.1953039
