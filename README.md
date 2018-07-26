# Regional Attention Based Deep Feature for Image Retrieval


**Trained weights**
```
>>mkdir weights
>>cd weights
>>wget --no-check-certificate https://sglab.kaist.ac.kr/RegionalAttention/weights.tar
>>tar -xvf weights.tar
>>cd ..
```



**Evaluation:**
```
>>cd datasets/evaluation
>>wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
>>g++ -O compute_ap.cpp -o compute_ap
>>cd ../..
```

**Oxford:**
```
>>cd datasets/Oxford
>>wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
>>tar -xzf oxbuild_images.tgz -C jpg
>>wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
>>tar -xzf gt_files_170407.tgz -C lab
>>cd ../..
```

**Paris**
```
>>cd datasets/Paris
>>mkdir tmp
# Images are in a different folder structure, need to move them around
>>wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
>>wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
>>tar -xzf paris_1.tgz -C tmp
>>tar -xzf paris_2.tgz -C tmp
>>find tmp -type f -exec mv {} jpg/ \;
>>rm -rf tmp
>>wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz
>>tar -xzf paris_120310.tgz -C lab
>>cd ../..
```

How to use the code
--------------------------------------------------------------------------------------------------
For testing, 
```
#With Oxford dataset
>> Python Measure_OxfordParis.py --dataset Oxford
#With Paris dataset
>> Python Measure_OxfordParis.py --dataset Paris
```
For training regional attention network,
```
>> Python Training_ImageNet.py
```
For learning PCA,
```
>> Python LearningPCA_landmark.py
```
