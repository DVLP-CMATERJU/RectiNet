# RectNet
## A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping

Capturing images of documents is one of the easiest
and most used methods of recording them. These images however,
being captured with the help of handheld devices, often lead to
undesirable distortions that are hard to remove. <br>We propose
a supervised Gated and Bifurcated Stacked U-Net module to
predict a dewarping grid and create a distortion free image
from the input. While the network is trained on synthetically
warped document images, results are calculated on the basis of
real world images. <br>The novelty in our methods exists not only in
a bifurcation of the U-Net to help eliminate the intermingling of
the grid coordinates, but also in the use of a gated network which
adds boundary and other minute line level details to the model.
The end-to-end pipeline proposed by us achieves state-of-the-art
performance on the DocUNet dataset after being trained on just
8 percent of the data used in previous methods.
