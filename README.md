# Devnagri OCR
This is a image classifier and template matching based OCR made for image processing project.

# About Dataset
- The dataset was created by extraction and manual annotation of thousands of characters from handwritten documents. Creator Name: Shailesh Acharya, Email: sailes437@gmail.com, Institution: University of North Texas, Cell: +19402200157 Creator Name: Prashnna Kumar Gyawali, Email: gyawali.prasanna@gmail.com, Institution: Rochester Institute of Technology

- Data Type: GrayScale Image The image dataset can be used to benchmark classification algorithm for OCR systems. The highest accuracy obtained in the Test set is 98.47%. Model Description is available in the paper.

- Image Format: .png Resolution: 32 by 32 Actual character is centered within 28 by 28 pixel, padding of 2 pixel is added on all four sides of actual character.

- S. Acharya, A.K. Pant and P.K. Gyawali â€œDeep Learning Based Large Scale Handwritten Devanagari Character Recognitionâ,In Proceedings of the 9th International Conference on Software, Knowledge, Information Management and Applications (SKIMA), pp. 121-126, 2015.

# Model Description

;)
> Our model using three models ##first model has two layers only 
Second model has two layers with middle layer having 128 nodes
And third model has three layers with second layer having 256 nodes, Third layer with 128 nodes
Final layer in each model has 46 nodes as there are 36 alphabet and 10 digits in our data
We are doing this to find whether accuracy increases or decreases with addition of layers

This text you see here is *actually* written in Markdown! To get a feel for Markdown's syntax, type some text into the left window and watch the results in the right.


### Installation



Install the dependencies and devDependencies and start the program.

```sh
$ python3
$ tensorflow
$ keras
```

For production environments...

```sh
$ # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#This is helpful in visualising matplotlib graphs
%matplotlib inline   
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras #The deep learning model we will use to train our dataset will make use of this
import tensorflow as tf
from PIL import Image as IMG #To read the image file
import os #To move through the folders and fetching the images
import matplotlib.pyplot as plt #To render Plots of our data
import sklearn.model_selection as smodel #To split the data for training and cross validation set
```

### Development


First Model:
```sh
model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(46, activation=tf.nn.softmax)
])
```

Second Model:
```sh
model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(46, activation=tf.nn.softmax)
])
```

Third Model:
```sh
model3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(46, activation=tf.nn.softmax)
])
```

# Result and Conclusion

After training 3 different models we got following accuracy for corresponding models :

-For training
```sh
Accuracy for  model1 is 82.39 %
Accuracy for  model2 is 99.7 %
Accuracy for  model3 is 97.1 %
```

-For testing
```sh
Accuracy for  model1 is 67.01 %
Accuracy for  model2 is 90.29 %
Accuracy for  model3 is 91.35 %
```


>So it's concluded that increasing the number of layers do increase the accuracy :)
