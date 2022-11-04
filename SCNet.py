# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:06:35 2019

@author: Dongdong Zhou @JYU 
"""
import keras
from keras.layers import Input, concatenate
#from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D,AveragePooling1D, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from sklearn import metrics
from keras.models import Model
from keras.utils import np_utils

import h5py
from keras.callbacks import ModelCheckpoint



seed = 1
np.random.seed(seed)

#import scipy.io as scio
sns.set()

DATA_FORMAT='channels_last'
# In[] load the data and label


hf = h5py.File('.../EDF_train_subject_data_20_fold3.h5', 'r')  # the format of data is .h5
train_x = hf.get('dataset_1')
train_x = np.asarray(train_x)
train_label = pd.read_csv('.../EDF_train_subjcet_label_20_fold3.csv') # the format of label is .csv

hf1 = h5py.File('.../EDF_test_subject_data_20_fold3.h5', 'r')
test_x = hf1.get('dataset_1')
test_x = np.asarray(test_x)
test_label = pd.read_csv('.../EDF_test_subjcet_label_20_fold3.csv')

train_y = np.asarray(train_label)
train_y = np_utils.to_categorical(train_y)
print('the shape of train label:', train_y.shape)


test_y = np.array(test_label)
test_y = np_utils.to_categorical(test_y)
print('the shape of test label:', test_y.shape)


train_x = np.expand_dims(train_x, axis = 2)
print('the shape of train set:', train_x.shape)
test_x = np.expand_dims(test_x, axis = 2)
print('the shape of test set:', test_x.shape)




fs = 100
DROPOUT = 0.5
NB_CLASS = 5
EPOCH = 30
batch_size = 64

filepath='.../1D_EDF_SCNet_subject_20_fold3.h5'  
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max') # Save the model with best performance



train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=seed)

def create_callbacks():
    early_stop = EarlyStopping(
    monitor         =       "val_loss",
    mode            =       "min",
    patience        =       3,
    verbose         =       1
    )
    return early_stop

# In[]
def max_aver_module1(x,padding='same'):
    path1 = MaxPooling1D(pool_size=3,strides=1,padding='same',data_format=DATA_FORMAT)(x)
    path2 = AveragePooling1D(pool_size=3,strides=1,padding='same',data_format=DATA_FORMAT)(x)
    
    return concatenate([path1, path2])

def plot_confusion_matrix(cm, labels_name, title):    # confusion matrix plot
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]   
    plt.imshow(cm, interpolation='nearest')   
    plt.title(title)    
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)   
    plt.yticks(num_local, labels_name)    
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
 
# In[] inception module 
def MC_block(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=1,activation='relu',use_bias=True,kernel_initializer='he_normal',bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=None):
    (branch1,branch2,branch3,branch4,branch5)=params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None


    # filter size: 1,
    pathway1 = Conv1D(filters=branch1[0],kernel_size=1,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer='he_normal',bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    # filter size: 1, 3
    pathway2 = Conv1D(filters=branch2[0],kernel_size=1,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway2 = Conv1D(filters=branch2[1],kernel_size=3,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)
  
    # filter size: 1,16
    pathway3 = Conv1D(filters=branch3[0],kernel_size=1,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway3 = Conv1D(filters=branch3[1],kernel_size=16,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)


    # filter size: 1,64
    pathway4 = Conv1D(filters=branch4[0],kernel_size=1,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway4 = Conv1D(filters=branch4[1],kernel_size=64,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)


    pathway5 = max_aver_module1(x,padding='same') # M-APooling1D 
    pathway5 = Conv1D(filters=branch5[0],kernel_size=1,strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway5)
    
    return concatenate([pathway1,pathway2,pathway3,pathway4,pathway5],axis=concat_axis)

#[pathway1,pathway2,pathway3, pathway4,pathway5]
def max_aver_module(x,padding='same'):
    path1 = MaxPooling1D(pool_size=3,strides=2,padding='same',data_format=DATA_FORMAT)(x)
    path2 = AveragePooling1D(pool_size=3,strides=2,padding='same',data_format=DATA_FORMAT)(x)
    
    return concatenate([path1, path2])

# In[]
def create_model():
    #Data format:tensorflow,channels_last;theano,channels_last
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(1,fs*90)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(fs*90,1)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=2
    else:
        raise Exception('Invalid Dim Ordering')

    print('the shape of img_input:',img_input.shape)
    x = Conv1D(128,128,strides=2,padding='same',activation = 'relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001),input_shape=(fs*90,1))(img_input)
    print('training now!!')
    x = BatchNormalization()(x)
    x = max_aver_module(x,padding='same')   
    x = MC_block(x,params=[(32,),(48,64),(64,96),(16,48),(32,)],concat_axis=CONCAT_AXIS)

    x = BatchNormalization()(x)
    x = max_aver_module(x,padding='same')

    x = Dropout(0.1)(x)
    x = MC_block(x,params=[(32,),(48,64),(64,96),(16,48),(32,)],concat_axis=CONCAT_AXIS)
    x = BatchNormalization()(x)
    x = max_aver_module(x,padding='same')
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(output_dim=NB_CLASS,activation='softmax')(x)

    return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


def check_print():
    # Create the Model
    x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT = create_model()

    # Create a Keras Model
    model = Model(input=img_input,output=[x])
#    model.summary()

    adam = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    print ('Model Compiled')
    return model

if __name__=='__main__':
    model = check_print()
    
    callbacks = create_callbacks()
    print('------------ Start Training ------------')  
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,mode='auto',  verbose = 1, factor=0.5, min_lr = 0.0000001)   
    cnn_model = model.fit(train_x,train_y, validation_data=(val_x , val_y),
                          batch_size=batch_size,
                          epochs=EPOCH,
                          callbacks=[learning_rate_reduction,checkpoint], # reduce the leaning rate  when validation accuracy shows no improvement within 3 epochs
                          shuffle=True)
    
    pred = model.predict(test_x, batch_size=batch_size, verbose=1)
    predicted = np.argmax(pred, axis=1)
    test_label_number = np.argmax(test_y, axis=1)
    report = classification_report(np.argmax(test_y, axis=1), predicted)
    cm = confusion_matrix(np.argmax(test_y, 1), predicted)
    print(report)
    print('Confusion matrix:',cm)
    
    Acc = metrics.accuracy_score(test_label_number,predicted) 
    print ('Test result:acc:%f'%(Acc))


