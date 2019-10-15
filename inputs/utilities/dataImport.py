#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:45:20 2018

@author: carsault
"""

#%%
import torch
import torch.utils.data as data_utils
from random import randint
from torch.utils import data
from utilities import chordUtil
import numpy as np
import pickle
import os
import errno

def createDatasetFull(name):
    X = []
    y = []
    with open(name, 'rb') as pickle_file:
        test = pickle.load(pickle_file)
    return test

def saveSetDecim(listIDs, root, alpha, dictChord, dictChordGamme, gamme, lenSeq, lenPred, Decim, folder, part, lab = False):
    Xfull = []
    yfull = []
    keyfull = []
    beatfull = []
    dictDat = {}
    try:
        os.mkdir(folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    minorKey = 0
    for track in listIDs:
        #print(track)
        beatInf = []
        # Open xlab
        xlab = open(root + track,"r")
        lines = xlab.read().split("\n")
        # Transform with one chord by beat
        chordBeat = []
        key = []
        # Initialize with N
        for i in range(lenSeq-1):
            beatInf.append(4) #if it's before start downbeat information is 4
            chordBeat.append(dictChord[chordUtil.reduChord('N', alpha)])
            key.append(dictChordGamme['N'])
        # Complete with chords in the file
        for i in range(len(lines)-1):
            line = lines[i+1].split(" ")
            downBeat = line[0].split(":")
            for j in range(int(line[2])):
                beatInf.append((int(downBeat[1])+j-1)%4) #get beat minus one, times j  
                chordBeat.append(dictChord[chordUtil.reduChord(line[4], alpha)])
                key.append(dictChordGamme[gamme[line[6]]])
        # Iterate over the track
        for start in range(len(chordBeat)-lenPred-lenSeq+1):
            if lab == False:
                X = torch.zeros(lenSeq, len(dictChord))
                for i in range(lenSeq):
                    X[i][chordBeat[start+i]] = 1
            else:
                X = torch.zeros(lenSeq)
                for i in range(lenSeq):
                    X[i] = chordBeat[start+i]
            # Get label
            if lab == False:
                y = torch.zeros(lenPred, len(dictChord))
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    y[i][chordBeat[start+lenSeq+i]] = 1
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
            else:
                y = torch.zeros(lenPred)
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    y[i] = chordBeat[start+lenSeq+i]
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
                    
            listX = []
            listy = []
            for i in Decim:
                u = []
                decimX = torch.chunk(X, int(lenSeq / i))
                for j in range(len(decimX)):
                    u.append(torch.sum(decimX[j], 0))
                u = torch.stack(u)
                listX.append(u)
                u = []
                decimy = torch.chunk(y, int(lenPred / i))
                for j in range(len(decimy)):
                    u.append(torch.sum(decimy[j], 0))
                u = torch.stack(u)
                listy.append(u)
            listX = torch.cat(listX, 0)
            listy = torch.cat(listy, 0)
            Xfull.append(listX)
            yfull.append(listy)
            beatfull.append(numBeat)
            keyfull.append(localKey)
            
    Xfull = torch.stack(Xfull)
    yfull = torch.stack(yfull)
    keyfull = torch.stack(keyfull)
    beatfull = torch.stack(beatfull)

    if lab == False:
        sauv = open(folder + '/' + part +".pkl","wb")  
        pickle.dump(data_utils.TensorDataset(Xfull, yfull, keyfull, beatfull),sauv)
    else:
        dictDat["X"] = Xfull
        dictDat["y"] = yfull
        dictDat["key"] = keyfull
        dictDat["beat"] = beatfull
        sauv = open(folder + '/' + part + ".pkl","wb")  
        pickle.dump(dictDat,sauv)

    sauv.close()
    print("number of minor songs in this dataset:" + str(minorKey))
