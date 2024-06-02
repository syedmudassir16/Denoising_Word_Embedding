# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:18:08 2021

@author: phvpavankumar
"""

import random 
random.seed(10)

from sklearn.svm import SVC
#from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from deepforest import CascadeForestClassifier


import warnings
warnings.filterwarnings('always')


#SVM
def SVectorMachine(X_train, Y_train, X_test, Y_test):
    svc = SVC( kernel='rbf',max_iter=1000,decision_function_shape='ovo',random_state=10)
    svc.fit(X_train,Y_train)
    pred_svc = svc.predict(X_test)
    acc = f'{100*accuracy_score(Y_test,pred_svc):.4f}%'
    prec = f'{100*precision_score(Y_test,pred_svc, average="macro"):.4f}%'
    f1 = f'{100*f1_score(Y_test,pred_svc, average="macro"):.4f}%'
    recall = f'{100*recall_score(Y_test,pred_svc, average="macro"):.4f}%'
    Result1 = [acc, prec, f1, recall]
    return Result1

#DecisionTree

def Dtree(X_train, Y_train, X_test, Y_test,deepth=None):
    #Dtree = DecisionTreeClassifier(criterion='entropy', random_state=10)
    Dtree = DecisionTreeClassifier(max_depth=deepth, random_state=10)
    Dtree.fit(X_train,Y_train)
    pred_Dtree = Dtree.predict(X_test)
    acc = f'{100*accuracy_score(Y_test,pred_Dtree):.4f}%'
    prec = f'{100*precision_score(Y_test,pred_Dtree, average="macro"):.4f}%'
    f1 = f'{100*f1_score(Y_test,pred_Dtree, average="macro"):.4f}%'
    recall = f'{100*recall_score(Y_test,pred_Dtree, average="macro"):.4f}%'
    Result2 = [acc, prec, f1, recall]
    return Result2

#RandomForest
def Rforest(X_train, Y_train, X_test, Y_test,deepth=None):
    #RFC = RandomForestClassifier(criterion='entropy', random_state=10)
    RFC = RandomForestClassifier(max_depth=deepth, random_state=10)
    RFC.fit(X_train,Y_train)
    pred_RFC = RFC.predict(X_test)
    acc = f'{100*accuracy_score(Y_test,pred_RFC):.4f}%'
    prec = f'{100*precision_score(Y_test,pred_RFC, average="macro"):.4f}%'
    f1 = f'{100*f1_score(Y_test,pred_RFC, average="macro"):.4f}%'
    recall = f'{100*recall_score(Y_test,pred_RFC, average="macro"):.4f}%'
    Result3 = [acc, prec, f1, recall]
    return Result3

#Navies base 
def NB(X_train, Y_train, X_test, Y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    pred_NB = gnb.predict(X_test)
    acc = f'{100*accuracy_score(Y_test,pred_NB):.4f}%'
    prec = f'{100*precision_score(Y_test,pred_NB, average="macro"):.4f}%'
    f1 = f'{100*f1_score(Y_test,pred_NB, average="macro"):.4f}%'
    recall = f'{100*recall_score(Y_test,pred_NB, average="macro"):.4f}%'
    Result4 = [acc, prec, f1, recall]
    return Result4

#KNN
def KNN(X_train, Y_train, X_test, Y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, Y_train)
    Pred_KNN = classifier.predict(X_test)
    acc = f'{100*accuracy_score(Y_test,Pred_KNN):.4f}%'
    prec = f'{100*precision_score(Y_test,Pred_KNN, average="macro"):.4f}%'
    f1 = f'{100*f1_score(Y_test,Pred_KNN, average="macro"):.4f}%'
    recall = f'{100*recall_score(Y_test,Pred_KNN, average="macro"):.4f}%'
    Result5 = [acc, prec, f1, recall]
    return Result5
    

#DeepForest
def DeepForest(X_train, Y_train, X_test, Y_test):
#     DeepFor = CascadeForestClassifier(criterion='entropy',random_state=10)#
    DeepFor = CascadeForestClassifier(random_state=10)
    DeepFor.fit(X_train,Y_train)
    pred_DeepFor = DeepFor.predict(X_test)
    acc = f'{100*accuracy_score(Y_test,pred_DeepFor):.4f}%'
    prec = f'{100*precision_score(Y_test,pred_DeepFor, average="macro"):.4f}%'
    f1 = f'{100*f1_score(Y_test,pred_DeepFor, average="macro"):.4f}%'
    recall = f'{100*recall_score(Y_test,pred_DeepFor, average="macro"):.4f}%'
    Result6 = [acc, prec, f1, recall]
    return Result6
