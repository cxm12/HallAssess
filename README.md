# HallAssess
"Assessing AI-Reconstructed Hallucinations in Fluorescence Microscopy Image"

### Requirements
* Python 3.7
* CUDA 11.4 and CUDNN 
* Packages: 
  
  torch            ==          1.10.0+cu113
  timm             ==          0.6.7
  basicsr          ==          1.4.2
  imageio          ==          2.13.3
  keras            ==          2.11.0
  numpy            ==          1.21.5
  opencv-python    ==          4.5.4.60
  Pillow           ==          9.0.1
  scikit-image     ==          0.19.2
  scipy            ==          1.7.3
  tifffile         ==          2021.11.2


### Prediction
The proposed HallAssess can assess the hallucination in AI-reconstructed fluorescence microscopy images without requiring a high-quality (HQ) image as the reference.

```
cd <directory of the .py file>
python main.py
```
Evaluating the hallucinations of super-resolution results from the `CCPs` test set:

setting task = 1 and testset = 'CCPs'

Replacing "FMIRdatapath" with the folder name containing AI-reconstructed images, such as super-resolved/denoised images, which need to be measured. 

Replacing "LQpath" with the folder name containing low-quality images (such as low-resolution images, noisy images), which are the input of the fluorescence microscopy image restoration models (such as super-resolution models, denoising models) for generating AI-reconstructed images. 


### Data
* The example images in the paper can be found in the `./example_data/` file.
  
The training and test data involved in the experiments are obtained through the data generation method introduced in the paper. The AI-reconstructed images can be obtained by applying an AI-based fluorescence microscopy image restoration method on the original low-quality (LQ) image. 

The imaging model is trained on publicly available datasets by taking the HR/clean images as input and the LR/noisy images as input: 

* The 3D denoising datasets can be downloaded from `https://publications.mpi-cbg.de/publications-sites/7207/`

* The SR dataset can be downloaded from `https://doi.org/10.6084/m9.figshare.13264793`

### Model
The pretrained imaging models in our HallAssess are put into `./experiment/`
