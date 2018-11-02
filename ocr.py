# This Python 3 environment comes with many helpful analytics libraries installed
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

def countfile(root_dir):
    '''This function will move through all directory and 
    count the no. of images in our training set'''
    count = 0
    parent_folders = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for characterfolder in parent_folders:
        if(os.path.isdir(characterfolder)):
            characterimage = [os.path.join(characterfolder,f) for f in os.listdir(characterfolder)]
            for image in characterimage:
                count += 1
    return count

def make_matrix(root_dir):
    '''This will make our feature matrix and label matrix used to train our model
    '''
    size = countfile(root_dir)
    X = np.zeros((size,32,32))
    Y = np.zeros((size,1),dtype='S140')
    Id = 0
    parent_folders = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for characterfolder in parent_folders:
        if(characterfolder):
            print(characterfolder)
            characterimage = [os.path.join(characterfolder,f) for f in os.listdir(characterfolder)]
            for image in characterimage:
                I = np.array(IMG.open(image))#This will open the image and parse the data as it's pixel values
                X[Id] = I #Used to assign the pixel data for each image
                filepathTokens=image.split('/') 
                Y[Id,] = str(filepathTokens[-2].split('_')[-1])
                Id += 1
    return X,Y

rootdir = "DevanagariHandwrittenCharacterDataset/Train"  #This is our directory inside which all images are present in different subfolders.
X,Y = make_matrix(rootdir)

#This will make a dictionary with keys as the labels and value as the numeric label we want to assign to each string label
Yclass = {}
for i in range(len(np.unique(Y)[:])):
    Yclass[np.unique(Y)[i]] = i
    
def yvectorize(dict,data):
    '''This will assign the numeric label to each string label in the label matrix'''
    return dict[data]
vect = np.vectorize(yvectorize)
Y = vect(Yclass,Y)

x_train,x_test,y_train,y_test = smodel.train_test_split(X,Y,test_size=0.3) #This will split the matrix into train and cross validation matrix

#to find the character name by it's label
def charactername(dic,label):
    for key, value in dic.items():    # for name, age in list.items():  (for Python 3.x)
        if(value == label):
            return (key)
print("Y shape",Y.shape)
print("X shape",X.shape)
#To print a random image and it's label from trainig set
rand = np.random.randint(1,100)
plt.figure()
plt.imshow(x_train[rand])
plt.colorbar()
plt.gca().grid(False)
plt.xlabel(charactername(Yclass,y_train[rand]))

"""let's plot 25 random train set images """
plt.figure(figsize=(10,10))
rand = np.random.randint(1,1000,25)
for i in range(len(rand)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(x_train[rand[i]], cmap=plt.cm.binary)
    plt.xlabel(charactername(Yclass,y_train[rand[i]]))
    
#Let's normalise the feature matrix 
"""The value of feature matrix are in range 0 - 255"""
x_train = (x_train-np.mean(x_train))/np.std(x_train)
x_test = (x_test-np.mean(x_test))/np.std(x_test)

"""Let's train our model using three models ##first model has two layers only 
Second model has two layers with middle layer having 128 nodes
And third model has three layers with second layer having 256 nodes, Third layer with 128 nodes
Final layer in each model has 46 nodes as there are 36 alphabet and 10 digits in our data
We are doing this to find whether accuracy increases or decreases with addition of layers"""
model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(46, activation=tf.nn.softmax)
])

model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(46, activation=tf.nn.softmax)
])


model3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(46, activation=tf.nn.softmax)
])
modellist = {'model1':model1,'model2':model2,'model3':model3}

"""Compiling our models"""
for k in modellist:
    modellist[k].compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
"""To note the value of epochs in corresponding model and train and test accuracy with that given model and epochs"""
epo = {'model1':[],'model2':[],'model3':[]}
trainacc = {'model1':[],'model2':[],'model3':[]}
testacc = {'model1':[],'model2':[],'model3':[]}
def trainandaccuracy(epochs,model,i):
    print("--"*20)
    print(i,epochs)
    model.fit(x_train,y_train, epochs=epochs)
    test_loss, test_acc = model.evaluate(x_test,y_test)
    train_loss, train_acc = model.evaluate(x_train,y_train)
    epo[i].append(epochs)
    trainacc[i].append(train_acc)
    testacc[i].append(test_acc)
    
"""To train models with epochs 500  with all the three model we have created"""
for i in modellist:
    trainandaccuracy(500,modellist[i],i)
    
"""For training accuracy"""    
for m in trainacc:
    accuracy = float(trainacc[m][0])
    accuracy = round(accuracy*100,2)
    print("Accuracy for ",m,"is",accuracy,"%")

"""For testing accuracy"""
for m in testacc:
    accuracy = float(testacc[m][0])
    accuracy = round(accuracy*100,2)
    print("Accuracy for ",m,"is",accuracy,"%")