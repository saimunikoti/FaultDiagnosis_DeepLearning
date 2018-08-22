#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 00:04:28 2018

@author: sai
"""
####################################### import different packages ######################################

import numpy as np
import keras as kr
from keras import layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation, Masking
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling1D
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt

#%%######################################## load the trained model #####################################

model=kr.models.load_model("weights.331-0.0000") # load the saved trained model

#%%####################################### load the data ################################################

x_test1= np.load("x_test1.npy") # load the file of .npy format
y_test1= np.load("y_test1.npy") # load the file of .npy format

#x_test1= sio.loadmat("Local address of the test file ") # load the file of .mat format


#%%####################################### predict from the model #####################################

y_pred1=model.predict(x_test1)

#%%############################### Evaluates the performance of the model ###############################

""" scores gives loss and accuracy of the model on test data
    confusion matrix
    precision ratio
    recall ratio
    F-score
    kapp_score
"""

array=np.array([1,2,3,4,5,6])
uniques, ids = np.unique(array, return_inverse=True)

y_a=uniques[y_test1.argmax(1)]
y_p=uniques[y_pred1.argmax(1)]

cnf_matrix = metrics.confusion_matrix(y_a, y_p) 

Precision=metrics.precision_score(y_a, y_p, average='macro')

Recall=metrics.recall_score(y_a, y_p, average='macro') 

F1score=metrics.f1_score(y_a, y_p, average='weighted')  

#%%#################################### Extract the intermediate layer output ##########################

from keras.models import Model

layer_name = 'Layer_name' # each layer name can be seen from the command "model.summary()"

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output) # model is the original trained model 

intermediate_output = intermediate_layer_model.predict(x_test)

#%%####################################### t-SNE for visualization ##################################################

from sklearn.manifold import TSNE

""" dimension of each test example is (2000,1) and for visualization we want the dimension to be (2,1).
    X_embedded  will store the transformed coordinates. 
"""

X_embedded = TSNE(n_components=2).fit_transform(intermediate_output) # n_components is the desired dimension 

#%%########################### Customize plots ####################################

# Stationary 2D data
import matplotlib 

matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)
plt.scatter(X1[0:500,0],X1[0:500,1], color='r', s=2*s, marker='o', alpha=1)
plt.scatter(X1[500:1000,0], X1[500:1000,1], color='b', s=2*s, marker='o', alpha=1) 
plt.scatter(X1[1000:1200,0], X1[1000:1200,1], color='g', s=2*s, marker='o', alpha=1)
plt.scatter(X1[1200:1500,0], X1[1200:1500,1], color='y', s=2*s, marker='o', alpha=1)
plt.scatter(X1[1500:1580,0], X1[1500:1580,1], color='m', s=2*s, marker='o', alpha=1)
plt.scatter(X1[1580:2080,0],X1[1580:2080,1], color='c', s=2*s, marker='o', alpha=1)


# non stationary data 

plt.scatter(X1[0:500,0],X1[0:500,1], color='r', s=2*s, marker='o', alpha=1)
plt.scatter(X1[500:1000,0], X1[500:1000,1], color='b', s=2*s, marker='o', alpha=1) 
plt.scatter(X1[1000:1500,0], X1[1000:1500,1], color='g', s=2*s, marker='o', alpha=1)
plt.scatter(X1[1500:2000,0], X1[1500:2000,1], color='y', s=2*s, marker='o', alpha=1)
plt.scatter(X1[2000:2500,0], X1[2000:2500,1], color='m', s=2*s, marker='o', alpha=1)
plt.scatter(X1[2500:3000,0],X1[2500:3000,1], color='c', s=2*s, marker='o', alpha=1)


plt.xlabel('Dimension 1',fontsize=25) 
plt.ylabel('Dimension 2',fontsize=25) 

plt.title('t-SNE for CNN Model (Variable Loading)',fontsize=25)

#plt.title('Features visualization for Stationary condition',fontsize=30)
#%%  plot scatter 3D data 

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=130) # figure specification

# Non stationary data in 3d plots

ax.scatter(X1[0:500,0],X1[0:500,1],X1[0:500,2], color='r', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[500:1000,0], X1[500:1000,1],X1[500:1000,2], color='b', s=2*s, marker='o', alpha=.8) 
ax.scatter(X1[1000:1500,0], X1[1000:1500,1],X1[1000:1500,2], color='g', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[1500:2000,0], X1[1500:2000,1],X1[1500:2000,2], color='y', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[2000:2500,0], X1[2000:2500,1],X1[2000:2500,2], color='m', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[2500:3000,0],X1[2500:3000,1], X1[2500:3000,2],color='c', s=2*s, marker='o', alpha=.8)

#ax.scatter(X1[0:1000,0], X1[0:1000,1],X1[0:1000,2], color='r', s=2*s, marker='o', alpha=.8)
#ax.scatter(X1[1000:2000,0], X1[1000:2000,1],X1[1000:2000,2], color='b', s=2*s, marker='o', alpha=.8)
#ax.scatter(X1[2000:3000,0], X1[2000:3000,1],X1[2000:3000,2], color='g', s=2*s, marker='o', alpha=.8)
#ax.scatter(X1[3000:4000,0], X1[3000:4000,1],X1[3000:4000,2], color='c', s=2*s, marker='o', alpha=.8)
#ax.scatter(X1[4000:5000,0], X1[4000:5000,1],X1[4000:5000,2], color='y', s=2*s, marker='o', alpha=.8)
#ax.scatter(X1[5000:6000,0], X1[5000:6000,1],X1[5000:6000,2], color='m', s=2*s, marker='o', alpha=.8)
 
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.title('3 dimension features visualization for Non Stationary Condition')
   
# stationary 3d plot
ax.scatter(X1[0:500,0], X1[0:500,1],X1[0:500,2], color='r', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[500:1000,0], X1[500:1000,1],X1[500:1000,2], color='b', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[1000:1200,0], X1[1000:1200,1],X1[1000:1200,2], color='g', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[1200:1500,0], X1[1200:1500,1],X1[1200:1500,2], color='y', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[1500:1580,0], X1[1500:1580,1],X1[1500:1580,2], color='c', s=2*s, marker='o', alpha=.8)
ax.scatter(X1[1580:2080,0], X1[1580:2080,1],X1[1580:2080,2], color='m', s=2*s, marker='o', alpha=.8)

# Legend in plot
import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='Healthy Condition') 
blue_patch = mpatches.Patch(color='blue', label='Inter Turn') 
green_patch = mpatches.Patch(color='green', label='Bearing Outerrace')
yellow_patch = mpatches.Patch(color='yellow', label='Bearing Genroughness')
magenta_patch = mpatches.Patch(color='magenta', label='Angular Misalignment')
cyan_patch = mpatches.Patch(color='cyan', label='Parallel Misalignment')

plt.legend(handles=[red_patch,blue_patch,green_patch,yellow_patch,magenta_patch,
                    cyan_patch], fontsize = 25)

red_patch = mpatches.Patch(color='red', label='C1') 
blue_patch = mpatches.Patch(color='blue', label='C2') 
green_patch = mpatches.Patch(color='green', label='C3')
yellow_patch = mpatches.Patch(color='yellow', label='C4')
magenta_patch = mpatches.Patch(color='magenta', label='C5')
cyan_patch = mpatches.Patch(color='cyan', label='C6')

plt.legend(handles=[red_patch,blue_patch,green_patch,yellow_patch,magenta_patch,
                    cyan_patch], fontsize = 25)
                             
loc='lower left'




