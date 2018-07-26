# Regional Attention Based Deep Feature for Image Retrieval


### Training datasets:

We use ImageNet dataset for training the regional attention network and Landmark dataset for learning PCA parameters of off-the-shelf Resnet101. 

**Trained weights**
```
mkdir weights
cd weights
wget --no-check-certificate https://sglab.kaist.ac.kr/RegionalAttention/weights.tar
tar -xvf weights.tar
cd ..
```


### Test datasets:

The code is prepared for testing with Oxford5k and Paris6k. 
Oxford105k and Paris106k can be easily expanded by adding flickr_100k images into the jpg folder existing in Paris6k and Oxford5k folders. 
**Evaluation:**
```
mkdir datasets
cd datasets
mkdir evaluation
cd evaluation
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
g++ -O compute_ap.cpp -o compute_ap
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

**Paris**
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

How to use the code
--------------------------------------------------------------------------------------------------
For testing, 
```
#With Oxford dataset
python Measure_OxfordParis.py --dataset Oxford
#With Paris dataset
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
