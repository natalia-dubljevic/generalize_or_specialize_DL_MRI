## Abstract
With the advent of deep-learning based MR image reconstruction, new questions are raised regarding best design choices. In particular, when using recent reconstruction models that have been developed for processing multi-channel MR data. A fundamental question is whether to first combine channels to improve model generalizability (i.e., a 'coil-combined' approach) or to keep channel processing separate to fully utilize multi-channel information (i.e., an 'all-coil' approach). In this work, we compare three popular architectures using coil-combined and all-coil styles on brain MR images. All-coil styles improved in-distribution performance, such as when reconstructing only presumed healthy individuals. Coil-combined designs better generalized to unseen data from patients with pathology.

## Code
This repository contains the source code for the following paper: 

Dubljevic N, Frayne R, Souza R.  Generalize or specialize? The effect of coil combining in deep learning based MR image reconstruction. In: 22nd International Symposium on Biomedical Imaging (ISBI), 2024 (accepted).

The code was developed using Python 3.10.9 and Pytorch 1.13.1. The .yml file can be used to recreate the conda environment used for this project. The trained model weights can be found [here](https://drive.google.com/drive/folders/140XH-KsaKnmirLDBl_IuLCz3EZ4SMoYa?usp=drive_link).


## Data
The data used to train and test the models is publicly available at as part of the [Calgary-Campinas dataset](https://sites.google.com/view/calgary-campinas-dataset/home). Specifically, the 12-channel raw dataset was used. Further model testing was performed on a private dataset from an ongoing study of brain tumours (glioblastoma).