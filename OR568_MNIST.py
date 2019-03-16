# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:39:42 2018

@author: savi0
"""

# load modules
import numpy as np 
import pandas as pd 
np.random.seed(1)
# get data
#Kfold_ds= pd.read_csv('C:/Users/savi0/OR568_Digit_savi/kfold_sample_sets.data')
test  = pd.read_csv('C:/Users/savi0/OR568_Digit_savi/test.csv')
train = pd.read_csv('C:/Users/savi0/OR568_Digit_savi/train.csv')
print(train.shape)
print(test.shape)
train.head()

#histogram shows the count of digits in the training data for each number
import matplotlib.pyplot as plt
plt.hist(train["label"])
plt.title("Frequency Histogram of Numbers in Training Data")
plt.xlabel("Number Value")
plt.ylabel("Frequency")

import math
# plot the first 25 digits in the training set. 
f, ax = plt.subplots(5, 5)
# plot some 4s as an example
for i in range(1,26):
    # Create a 1024x1024x3 array of 8 bit unsigned integers
    data = train.iloc[i,1:785].values #this is the first number
    nrows, ncols = 28, 28
    grid = data.reshape((nrows, ncols))
    n=math.ceil(i/5)-1
    m=[0,1,2,3,4]*5
    ax[m[i-1], n].imshow(grid)

## normalize data ##
label_train=train['label']
train=train.drop('label', axis=1)
#print(label_train)

train = train / 255
test = test / 255
train['label'] = label_train
#print(train['label'])
#PCA

#from sklearn import decomposition
from sklearn import decomposition
from sklearn.decomposition import PCA
pca = PCA(n_components=785)
pca.fit(train)
plt.plot(pca.explained_variance_ratio_)
pca = decomposition.PCA(n_components=50) #use first 3 PCs (update to 100 later)
pca.fit(train.drop('label', axis=1))
PCtrain = pd.DataFrame(pca.transform(train.drop('label', axis=1)))
PCtrain['label'] = train['label']

#decompose test data
#pca.fit(test)
PCtest = pd.DataFrame(pca.transform(test))
#Neural Net

from sklearn.neural_network import MLPClassifier
y = PCtrain['label'][0:20000]
X=PCtrain.drop('label', axis=1)[0:20000]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(3500,), random_state=1)
clf.fit(X, y)

from sklearn import  metrics
#accuracy and confusion matrix on train

predicted = clf.predict(PCtrain.drop('label', axis=1)[20001:42000])
expected = PCtrain['label'][20001:42000]
print("Classification report for classifier %s:\n%s\n"% (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#On Test Set
#print("Classification report for classifier %s:\n%s\n"% (clf,metrics.classification_report(test['label'],Results_NN))


Results_NN= clf.predict(PCtest)
output = pd.DataFrame(Results_NN, columns =['Label'])
output.reset_index(inplace=True)
output.rename(columns={'index': 'ImageId'}, inplace=True)
output['ImageId']=output['ImageId']+1
output.to_csv('output.csv', index=False)


#run predict_proba
import scikitplot as skplot
import pandas as pd
import numpyt as np
import matplotlip.pylot as plt
porbs=clf.predict_proba(PCtest)
print(porbs)
skplot.metrics.plot_roc_curve(Results_NN,porbs)
plt.show()

import numpy
from pyearth import Earth
from matplotlib import pyplot
model = Earth()
model.fit(X,y)



###################################

#Random Forest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
#k_fold = KFold(n_splits=3)
#https://stackoverflow.com/questions/47949233/random-forest-python-sklearn

clf2 = RandomForestClassifier(n_estimators =50)
clf2.fit(X, y)


predicted2 = clf2.predict(PCtrain.drop('label', axis=1)[20001:42000])
expected2 = PCtrain['label'][20001:42000]

print("Classification report for Random Forest %s:\n%s\n"% (clf2, metrics.classification_report(expected2, predicted2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected2, predicted2))

Results_RF=clf2.predict(PCtest)

print(clf2.feature_importances_)
np.savetxt('resultN.csv', 
           np.c_[range(1,len(PCtest)+1),Results_RF], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

#####################################

#Convoulution NN
import Theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

NNet= NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )

NNet.fit(X, y)



