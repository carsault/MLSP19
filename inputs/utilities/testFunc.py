#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:11:28 2019

@author: carsault
"""

#%%
import math
import torch
import numpy as np
from utilities import distance 
from utilities import C2V
from utilities import chordUtil as utils
from utilities import loss
from utilities.loss import *
#from aiayn import aiayn
#from aiayn.aiayn import *
import torch.nn.functional as F

def accuracy(output, label, lenSeq, lenPred):
    correct = 0
    total = output.size()[0]*lenPred
    for i in range(output.size()[0]):
        for j in range(lenPred):
            correct += (output[i][j].max(0)[1] == label[i][j].max(0)[1]).item()
    return correct, total

def accuracyTransf(output, label, lenSeq, lenPred):
    correct = 0
    total = output.size()[0]
    for i in range(output.size()[0]):
        correct += (output[i].max(0)[1] == label[i].max(0)[1]).item()
    return correct, total

def repeatAccuracy(output, label, lenSeq, lenPred):
    correct = 0
    total = output.size()[0]*lenPred
    for i in range(output.size()[0]):
        for j in range(lenPred):
            correct += (output[i][lenSeq-1].max(0)[1] == label[i][j].max(0)[1]).item()
    return correct, total

def accuracyS2S(output, label, lenSeq, lenPred):
    correct = 0
    total = output.size()[0]*lenPred
    for i in range(output.size()[0]):
        for j in range(lenPred):
            correct += (output[i][j] == label[i][j]).item()
    return correct, total

def repeatAccuracyS2S(output, label, lenSeq, lenPred):
    correct = 0
    total = output.size()[0]*lenPred
    for i in range(output.size()[0]):
        for j in range(lenPred):
            correct += (output[i][lenSeq-1] == label[i][j]).item()
    return correct, total

def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

def musicalDist(output, label, lenSeq, lenPred, dist, tf_mappingR):
    totDist = 0
    #tf_mappingR = torch.tensor(tf_mappingR).float()
    for i in range(output.size()[0]):
        for j in range(lenPred):
            #totDist += np.dot(np.dot(output[i][j].detach().numpy(), tf_mappingR),label[i][j].detach().numpy())
            totDist += torch.dot(torch.matmul(output[i][j], tf_mappingR),label[i][j])
    return totDist

def computeMat(dictChord, dist):
    if dist == 'tonnetz':
        tf_mappingR = distance.tonnetz_matrix((invert_dict(dictChord),invert_dict(dictChord)))
        #tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
        #tf_mappingR = 1./ tf_mappingR
        #tf_mappingR = (tf_mappingR) / np.max(tf_mappingR)
        #print(tf_mappingR)
    elif dist == 'euclidian':
        tf_mappingR = distance.euclid_matrix((invert_dict(dictChord),invert_dict(dictChord)))
        #tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
        #tf_mappingR = 1./ tf_mappingR
        #tf_mappingR = (tf_mappingR) / np.max(tf_mappingR) 
    else:
        raise ValueError('Dist function named '+dist+' not defined')
    return tf_mappingR