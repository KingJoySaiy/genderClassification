

# Gender Classification

Implementation of face-gender-classification by ResNet-50 (machine learning). 

## 1. Requirement

* Python 3.7
* torch1.0.0
* torchvision0.2.2

## 2. data

* **train.csv** - training set, including 2 columns( id: name of the face image, label: gender label, 0 means male, 1 means female)
* **test.csv** - test set, only includes a column of id, that is, the numbers of all face images in the test set. No gender label in the test set
* **train/train/** - folder of all training images, the extension is jpg, and each name is the same as the id in train.csv
* **test/test/** - folder of all test images, the extension is jpg, and each name is the same as the id name in test.csv
* **sampleSubmit.csv** - a sample of the submitted file including 2 columns( id: name of the test-face-image, label: gender label output by the model, 0 means male, 1 means female)
* to download full data [click here](https://www.kaggle.com/c/jiangnan2020/data) 

## 3. Get started

1. set some constants in `genderClassification/constant/constPath`
2. run `python train.py`, model will be saved in `genderClassification/savedModel`
3. run `python predict.py [name of savedModel]`, default is `model-1.pkl`

