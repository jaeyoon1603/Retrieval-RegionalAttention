# Regional Attention Based Deep Feature for Image Retrieval

**Regional Attention Based Deep Feature for Image Retrieval (BMVC 2018)**
*Jaeyoon Kim and Sung-Eui Yoon*

The project page is in https://sglab.kaist.ac.kr/RegionalAttention/

Most of the code, such as evaluation code, R-MAC code, etc, are built upon [Deep Image Retrieval (ECCV, 2016)](https://github.com/figitaki/deep-retrieval).

Dependency
--------------------------------------------------------------------------------------------------
`Pytorch`, 0.3.0 required.


Build datasets
--------------------------------------------------------------------------------------------------
### Test datasets
The code is prepared for testing with Oxford5k and Paris6k. 
Oxford105k and Paris106k can be easily expanded by adding flickr_100k images into the jpg folder existing in Paris6k and Oxford5k folders. 

**Trained weights:**
```
mkdir weights
cd weights
wget --no-check-certificate https://sglab.kaist.ac.kr/RegionalAttention/weights.tar
tar -xvf weights.tar
cd ..
```

**Evaluation:**
```
mkdir datasets
cd datasets
mkdir evaluation
cd evaluation
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
sed -i '6i#include <cstdlib>' compute_ap.cpp # Add cstdlib, as some compilers will produce an error otherwise
g++ -o compute_ap compute_ap.cpp
cd ../..
```

**Oxford:**
```
cd datasets
mkdir Oxford
cd Oxford
mkdir jpg lab
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
tar -xzf oxbuild_images.tgz -C jpg
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
tar -xzf gt_files_170407.tgz -C lab
cd ../..
```

**Paris:**
```
cd datasets
mkdir Paris
cd Paris
mkdir tmp jpg lab
# Images are in a different folder structure, need to move them around
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
tar -xzf paris_1.tgz -C tmp
tar -xzf paris_2.tgz -C tmp
find tmp -type f -exec mv {} jpg/ \;
rm -rf tmp
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz
tar -xzf paris_120310.tgz -C lab
cd ../..
```

### Training datasets

We use ImageNet dataset for training the regional attention network and Landmark dataset for learning PCA parameters of off-the-shelf Resnet101. But, we do not publish the code for learning the PCA since Landmark dataset continues to be harmed by broken URLs of images. As an alternative, you can use our uploaded PCA weights.

**ImageNet**
```
cd datasets
mkdir ImageNet
cd ImageNet
mkdir train val
cd train
mkdir data
#####Need to download 'ILSVRC2012_img_train.tar' in http://www.image-net.org/ and move it to this directory#####
tar -xzf ILSVRC2012_img_train.tar -C data
cd data
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
cd val
mkdir data
#####Need to download 'ILSVRC2012_img_val.tar' in http://www.image-net.org/ and move it to this directory#####
tar -xzf ILSVRC2012_img_val.tar -C data
```



How to use the code
--------------------------------------------------------------------------------------------------
For testing, 
```
#When testing with Oxford dataset
python Measure_OxfordParis.py --dataset Oxford
#When testing with Paris dataset
python Measure_OxfordParis.py --dataset Paris
```
For training regional attention network,
```
python Training_ImageNet.py
```
For learning PCA,
```
python LearningPCA_landmark.py
```
Citation
--------------------------------------------------------------------------------------------------
Please cite this paper if you use this code in an academic publication.
```
@InProceedings{retrieval:BMVC:2018,
  author  = {Jaeyoon Kim and Sung-Eui Yoon},
  title   = {Regional Attention Based Deep Feature for Image Retrieval},
  booktitle = {Proc. British Machine Vision Conference (BMVC 2018)},
  address = {Newcastle, England},
  year = {2018},
  pages = {},
  volume  = {},
}
```
