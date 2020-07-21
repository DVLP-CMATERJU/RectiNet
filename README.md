# RectiNet
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-gated-and-bifurcated-stacked-u-net-module/ssim-on-docunet)](https://paperswithcode.com/sota/ssim-on-docunet?p=a-gated-and-bifurcated-stacked-u-net-module)

## A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping

Capturing images of documents is one of the easiest
and most used methods of recording them. These images however,
being captured with the help of handheld devices, often lead to
undesirable distortions that are hard to remove. We propose
a supervised Gated and Bifurcated Stacked U-Net module to
predict a dewarping grid and create a distortion free image
from the input. While the network is trained on synthetically
warped document images, results are calculated on the basis of
real world images. The novelty in our methods exists not only in
a bifurcation of the U-Net to help eliminate the intermingling of
the grid coordinates, but also in the use of a gated network which
adds boundary and other minute line level details to the model.
The end-to-end pipeline proposed by us achieves state-of-the-art
performance on the DocUNet dataset after being trained on just
8 percent of the data used in previous methods.

---

### Training the model
- Directory Structure:
```
.
+-- train.py
+-- dataset.py
+-- model.py
+-- plot_me.py
+-- predict.py
+-- data_gen
|   +-- image
|   +-- label
|   +-- image_test
```
- Run:
`python3 train.py --batch-size 16`
- For custom location of training data:
`python3 train.py --batch-size 16 --data-path PATH_TO_DATA`
- For more parameters:
` python3 train.py -help`



### Dense Grid Prediction and Image Unwarp
- In same directory:
` mkdir save`
- For predicting single image:
` python3 predict.py --save-path save --img-path IMAGE_PATH --model-path SAVED_MODEL_PATH --multi=False`
- For predicting many image in a folder:
` python3 predict.py --save-path save --img-path IMAGE_FOLDER_PATH --model-path SAVED_MODEL_PATH --multi=True`
- For more parameters:
` python3 predict.py -help`



