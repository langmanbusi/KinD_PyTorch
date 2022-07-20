# KinD_PyTorch
A PyTorch Implementation of KinD 

# Kindling the Darkness A Practical Low-light Image Enhancer, MM'19 
Unofficial PyTorch code for the paper - (MM 2019)Kindling the Darkness A Practical Low-light Image Enhancer 

Yonghua Zhang, Jiawan Zhang, Xiaojie Guo

The offical Tensorflow code is available [here](https://github.com/zhangyhuaee/KinD). 

Please ensure that you cite the paper if you use this code:
```
@inproceedings{zhang2019kindling,
 author = {Zhang, Yonghua and Zhang, Jiawan and Guo, Xiaojie},
 title = {Kindling the Darkness: A Practical Low-light Image Enhancer},
 booktitle = {Proceedings of the 27th ACM International Conference on Multimedia},
 series = {MM '19},
 year = {2019},
 isbn = {978-1-4503-6889-6},
 location = {Nice, France},
 pages = {1632--1640},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3343031.3350926},
 doi = {10.1145/3343031.3350926},
 acmid = {3350926},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {image decomposition, image restoration, low light enhancement},
}
```
### Training(LOL dataset)
Please download the training and testing datasets from [here](https://daooshee.github.io/BMVC2018website/). 
```
Data folder should like:
-- data_name(Eg. LOL)
  -- train
    -- low
    -- high
 -- test
    -- low
    -- high
```


And just run 
```
$ python train.py \
```


### Testing
For sample testing/prediction, you can run-
```
$ python predict.py
```

There is a pre-trained checkpoint available in the repo. You may use it for sample testing or create your own after training as needed. The results are generated (by default) for the data present in `./data/test/low/` folder, and the results are saved (by default) in `./results/test/low/` folder. 
Noticed that ckpt of Restore is in release, please download from latest release.

### Results FYI:
* MyKinD：**MSE** = 0.899  **SSIM** = 0.798 **PSNR** = 19.89 **LPIPS** =  0.138
	* Net：MyKinD
	* train dataset：LOL，test dataset：LOL
	* Decom: 1000, lr=0.0001, batchsize = 16，patchsize = 96
	* Restore: 1000, lr = 0.0001, batchsize = 4，patchsize = 384
	* Relight: 1000, lr = 0.0001, batchsize = 16，patchsize = 96

