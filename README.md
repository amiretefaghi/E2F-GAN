**E2F-GAN: Eyes-to-Face Inpainting via Edge-Aware Coarse-to-Fine GANs**
=======================================================================
This is the Tensorflow 2.0 implementation of paper 'E2F-GAN: Eyes-to-Face Inpainting via Edge-Aware Coarse-to-Fine GANs' which is submitted to IEEE Access journal.
After publication of our paper, the main code will be inserted.

**Introduction**
------------------------------------------------------------------------------------------------
This paper proposed a novel GAN-based deep learning model called Eyes-to-Face GAN (E2F-GAN) which includes two main modules: a coarse module and a refinement module. The coarse module along with an edge predictor module attempts to extract all required features from a periocular region and to generate a coarse output which will be refined by a refinement module. Additionally, a dataset of eyes-to-face synthesis has been generated based on the public face dataset called CelebA-HQ for training and testing. Thus, we perform both qualitative and quantitative evaluations on the generated dataset. Experimental results demonstrate that our method outperforms previous learning-based face inpainting methods and generates realistic and semantically plausible images. 

![image](E2F.bmp)

Prerequisites
---------------------------------
* Python 3.7
* Tensorflow 2.0
* NVIDIA GPU + CUDA cuDNN

Dataset
---------------------------------
We conduct all experiments on our generated dataset called E2Fdb extracted from the well-known [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset. To extract the periocular region from each face image, the images are reshaped to size  256 ×256 and then by utilizing a landmark detector , eyes are detected. Doing this, M and I_m are produced for each image. Moreover, we removed misleading samples including those eyes covered by sunglasses or faces that have more than 45 degrees in one angle (roll, pitch, yaw) leading to hiding one of the eyes by using WHENet algorithms. Finally, the total number of 
samples is 24,554 among which 22,879 will be used for the training process and the rest, which is 1,685 images, for the test.

