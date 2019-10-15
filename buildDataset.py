#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:42:27 2019

@author: carsault
"""
#%%
import argparse
import os
from utilities import chordUtil
from utilities.chordUtil import *
from sklearn.model_selection import train_test_split
from utilities import dataImport
from utilities.dataImport import *
from utilities import util
from utilities.util import *
#%%
"""
###################

Argument parsing

###################
"""
parser = argparse.ArgumentParser(description='Hierarchical Latent Space')
# General
parser.add_argument('--rootname',   type=str,   default='inputs/jazz_xlab/',    help='name of the data folder')
parser.add_argument('--dataFolder',   type=str,   default='testpremier2',    help='name of the data folder')
parser.add_argument('--random_state',   type=int,   default=2,    help='seed for the random train/test split')
parser.add_argument('--alpha',      type=str,   default='a0',                            help='type of alphabet')
parser.add_argument('--lenSeq',      type=int,   default= 8,                            help='length of input sequence')
parser.add_argument('--lenPred',      type=int,   default= 8,                            help='length of predicted sequence')
parser.add_argument('--decimList', nargs="+",     type=int,   default=[1, 2, 4],                            help='list of decimations (default: [1])')
parser.add_argument('--label',     type=util.str2bool,   default=False,                                 help='get lab (True) or one hot vector (False) (default: False)')

args = parser.parse_args()
print(args)
#%%

# List files
filenames = os.listdir(args.rootname)
filenames.remove(".DS_Store")
filenames.sort()

str1 = ''.join(str(e) for e in args.decimList)

args.dataFolder = args.alpha + "_" + str(str1) + "_" + str(args.random_state)
print(args.dataFolder)

# Get chord reduction
dictChord, listChord = chordUtil.getDictChord(eval(args.alpha))
dictKey, listKey = chordUtil.getDictKey()
#%%
# Create datasets
files_train ,files_test = train_test_split(filenames,test_size=0.4,random_state=args.random_state)
files_test ,files_valid = train_test_split(files_test,test_size=0.5,random_state=args.random_state)

dataset_train = dataImport.saveSetDecim(files_train, args.rootname, args.alpha, dictChord, dictKey, gammeKey, args.lenSeq, args.lenPred, args.decimList, "datasets/" + args.dataFolder, "/train", args.label)
dataset_valid = dataImport.saveSetDecim(files_valid, args.rootname, args.alpha, dictChord, dictKey, gammeKey, args.lenSeq, args.lenPred, args.decimList, "datasets/" + args.dataFolder, "/valid", args.label)
dataset_test = dataImport.saveSetDecim(files_test, args.rootname, args.alpha, dictChord, dictKey, gammeKey, args.lenSeq, args.lenPred, args.decimList, "datasets/" + args.dataFolder, "/test", args.label)

print("Dataset saved")
#%% test

import pickle
u = "249"
with open("datasets/" + "a0_1_123456" + "/train.pkl", 'rb') as pickle_file:
    test = pickle.load(pickle_file)
#print(test['X'])
#print(test['y'])

