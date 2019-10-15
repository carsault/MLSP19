#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:51:53 2019

@author: carsault
"""

#%%
import pickle
import torch
from utilities import chordUtil
from utilities.chordUtil import *
from utilities import testFunc
from utilities.testFunc import *
from utilities import distance
from utilities.distance import *
#from ACE_Analyzer import ACEAnalyzer
#from ACE_Analyzer.ACEAnalyzer import *
import numpy as np

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor

foldName = 'modelSave190520'
modelType = ["mlpDecim","lstmDecim","mlpDecimFamily"]
randm = [1, 2, 3, 4, 5]
alpha = ['a0','a2','a5']
dictKey, listKey = chordUtil.getDictKey()
correctDown = [0]*4
correctPos = [0]*8
totalDown = [0]*4
totalPos = [0]*8
musicalDist2 = 0
musicalDist4 = 0
dictFinalRes = {}
#%%
def perplexityAccumulate(res, y, sPrev, nby ,one_hot = False):
    s = 0
    oneHot = []
    if one_hot == True:
        for i in range(len(y)):
            oneHot.append(y[i].index(max(y[i])))
    else:
        oneHot = y
    for r, t in zip(res,oneHot):
        s += np.log2(r[t])

    sPrev += s
    nby += len(y)
    return sPrev, nby

def perplexityCompute(s,y):
    s /= -y
    return 2 ** s

#%%
def perplexity(res, y, one_hot = False):
    s = 0
    oneHot = []
    if one_hot == True:
        for i in range(len(y)):
            oneHot.append(y[i].index(max(y[i])))
    else:
        oneHot = y
    for r, t in zip(res,oneHot):
        s += np.log2(r[t])

    s /= -len(y)
    return 2 ** s
#%%
for alph in alpha:
    for model in modelType:
        print("Start test with " + model + " on alphabet " + alph)
        perp = []
        rank = []
        totRank = []
        #Analyzer = ACEAnalyzer()
        dictChord, listChord = chordUtil.getDictChord(eval(alph))
        distMat = testFunc.computeMat(dictChord, "euclidian")
        #distMat = torch.Tensor(distMat).to(device,non_blocking=True)
        #vectMat = distance.computeEuclideanPitchVectMat(dictChord)
        #sumPred2 = [0]*len(listChord)
        #sumPred4 = [0]*len(listChord)
        #sumTarg2 = [0]*len(listChord)
        #sumTarg4 = [0]*len(listChord)
        #perp = 0
        #sPrev = 0
        #nby = 0
        totAcc = []
        totAcc2 = []
        totAcc4 = []
        totAccDownbeat = []
        totAccPos = []
        totAccBeatPos = []
        totDist = []
        predTotDist = []
        totDistPred = []
        totDist2a = []
        totDist4a = []        
        totDist2b = []
        totDist4b = []
        totDist2c = []
        totDist4c = []
        for rand in randm:
            #sumPred2 = torch.zeros(len(listChord)).to(device,non_blocking=True)
            #sumPred4 = torch.zeros(len(listChord)).to(device,non_blocking=True)
            #sumTarg2 = torch.zeros(len(listChord)).to(device,non_blocking=True)
            #sumTarg4 = torch.zeros(len(listChord)).to(device,non_blocking=True)
            '''
            sumPred2ab = np.zeros(len(listChord))
            sumTarg2ab = np.zeros(len(listChord))
            sumPred4ab = np.zeros(len(listChord))
            sumTarg4ab = np.zeros(len(listChord))
            sumPred2c = np.zeros(len(listChord))
            sumTarg2c = np.zeros(len(listChord))
            sumPred4c = np.zeros(len(listChord))
            sumTarg4c = np.zeros(len(listChord))
            '''
            correctDown = [0]*4
            correctPos = [0]*8
            totalDown = [0]*4
            totalPos = [0]*8
            correctBeatPos = np.zeros((4,8))
            totalBeatPos = np.zeros((4,8))
            accBeatPos = np.zeros((4,8))
            musicalDist = 0
            predMusicalDist = 0
            acc2correct = 0
            acc4correct = 0
            musicalDist2a = 0
            musicalDist4a = 0
            musicalDist2b = 0
            musicalDist4b = 0
            musicalDist2c = 0
            musicalDist4c = 0
            zerr = np.zeros(len(dictChord))
            total = 0
            acc = 0
            acc2 = 0
            acc4 = 0
            correct = 0
            dataFolder = alph + "_124_" + str(rand)
            if model is "mlpDecimFamily":
                modelName = dataFolder + "_124_" + model
            else:
                modelName = dataFolder + "_1_" + model
            with open(foldName + '/' + modelName + '/' + "probVect_" + modelName + "_test.pkl", 'rb') as fp:
                dictDat = pickle.load(fp)
                dictDat["X"] = dictDat["X"].cpu().numpy()
                dictDat["y"] = dictDat["y"].cpu().numpy()
            musicalDist = np.sum(np.matmul(dictDat["X"], distMat) * dictDat["y"])
            for i in range(len(dictDat["X"])):
                #pred = dictDat["X"][i].index(max(dictDat["X"][i]))
                #tgt = dictDat["y"][i].index(max(dictDat["y"][i]))
                #pred = dictDat["X"][i].max(0)[1]
                #tgt = dictDat["y"][i].max(0)[1]
                pred = np.argmax(dictDat["X"][i])
                tgt = np.argmax(dictDat["y"][i])
                # rank of the chords
                #seq = sorted(dictDat["X"][i].tolist(), reverse=True)
                #index = [seq.index(v) for v in dictDat["X"][i].tolist()]
                #rank.append(index[tgt.item()] + 1)
                rank.append(len(np.where(dictDat["X"][i] > dictDat["X"][i][tgt])[0]) + 1)
                #print(rank)
                # ACE analizer
                #Analyzer.compare(chord = listChord[pred], target = listChord[tgt], key = listKey[dictDat["key"][i]], base_alpha = eval(alph) , print_comparison = False)
                # number of samples
                total += 1
                totalDown[dictDat["beat"][i]] += 1
                totalPos[dictDat["pos"][i]] += 1
                totalBeatPos[dictDat["beat"][i],dictDat["pos"][i]] += 1
                # Accuracy:
                if pred == tgt:
                    correct += 1
                    correctDown[dictDat["beat"][i]] += 1
                    correctPos[dictDat["pos"][i]] += 1
                    correctBeatPos[dictDat["beat"][i],dictDat["pos"][i]] += 1
                #musicalDist += np.matmul(dictDat["X"][i], distMat)[tgt]
                predMusicalDist += distMat[pred][tgt]
                '''
                # Multiscale musical distance
                #sumPred2 = sumPred2.add(dictDat["X"][i])
                #sumTarg2 = sumTarg2.add(dictDat["y"][i])
                #sumPred2c += dictDat["X"][i]
                #sumTarg2c += dictDat["y"][i]
                sumPred2ab[pred] += 1
                sumTarg2ab[tgt] += 1
                if i%2 == 1:
                    #musicalDist2 += np.dot(np.matmul(sumPred2, distMat),sumTarg2)
                    #sumPred2 = np.matmul(sumPred2, vectMat)
                    #sumTarg2 = np.matmul(sumTarg2, vectMat)
                    #musicalDist2 += np.linalg.norm(sumPred2-sumTarg2)
                    #musicalDist2a += np.linalg.norm(sumPred2ab-sumTarg2ab, ord = 1)
                    #musicalDist2b += np.dot(np.matmul(np.maximum(sumPred2ab-sumTarg2ab,zerr),distMat),np.maximum(sumTarg2ab-sumPred2ab,zerr))
                    #musicalDist2c += np.dot(np.matmul(sumPred2c, distMat),sumTarg2c)
                    acc2correct += 2 - sum(np.maximum(sumPred2ab-sumTarg2ab,0))
                    sumPred2ab = np.zeros(len(listChord))
                    sumTarg2ab = np.zeros(len(listChord))
                    sumPred2c = np.zeros(len(listChord))
                    sumTarg2c = np.zeros(len(listChord))
                    #musicalDist2 += torch.dot(torch.matmul(sumPred2, distMat),sumTarg2)
                    #sumPred2 = torch.zeros(len(listChord)).to(device,non_blocking=True)
                    #sumTarg2 = torch.zeros(len(listChord)).to(device,non_blocking=True)
                #sumPred4c += dictDat["X"][i]
                #sumTarg4c += dictDat["y"][i]
                sumPred4ab[pred] += 1
                sumTarg4ab[tgt] += 1
                #sumPred4 = sumPred4.add(dictDat["X"][i])
                #sumTarg4 = sumTarg4.add(dictDat["y"][i])
                if i%4 == 3:
                    #musicalDist4 += sumPred4*distMat*sumTarg4
                    #musicalDist4 += torch.dot(torch.matmul(sumPred4, distMat),sumTarg4)
                    #sumPred4 = torch.zeros(len(listChord)).to(device,non_blocking=True)
                    #sumTarg4 = torch.zeros(len(listChord)).to(device,non_blocking=True)      
                    #musicalDist4 += np.dot(np.matmul(sumPred4, distMat),sumTarg4)
                    #sumPred4 = np.matmul(sumPred4, vectMat)
                    #sumTarg4 = np.matmul(sumTarg4, vectMat)
                    #musicalDist4 += np.linalg.norm(sumPred4-sumTarg4)
                    #musicalDist4a += np.linalg.norm(sumPred4ab-sumTarg4ab, ord = 1)
                    #musicalDist4b += np.dot(np.matmul(np.maximum(sumPred4ab-sumTarg4ab,zerr),distMat),np.maximum(sumTarg4ab-sumPred4ab,zerr))
                    #musicalDist4c += np.dot(np.matmul(sumPred4c, distMat),sumTarg4c)
                    acc4correct += 4 - sum(np.maximum(sumPred4ab-sumTarg4ab,0))
                    sumPred4ab = np.zeros(len(listChord))
                    sumTarg4ab = np.zeros(len(listChord))
                    sumPred4c = np.zeros(len(listChord))
                    sumTarg4c = np.zeros(len(listChord))
                # perplexity
                '''
            #sPrev, nby = perplexityAccumulate(dictDat["X"].tolist(), dictDat["y"].tolist(), sPrev, nby, True)
            perp.append(perplexity(dictDat["X"].tolist(), dictDat["y"].tolist(), True))
            totRank.append(np.mean(rank))
            rank = []
            acc = correct/total
            acc2 = acc2correct/total
            acc4 = acc4correct/total
            accDownbeat = [int(b) / int(m) for b,m in zip(correctDown, totalDown)]
            accPos = [int(b) / int(m) for b,m in zip(correctPos, totalPos)]
            for i in range(len(totalBeatPos)):
                accBeatPos[i] = [int(b) / int(m) for b,m in zip(correctBeatPos[i], totalBeatPos[i])]
            '''    
            totAcc.append(acc)
            totAcc2.append(acc2)
            totAcc4.append(acc4)
            totAccDownbeat.append(accDownbeat)
            totAccPos.append(accPos)
            totAccBeatPos.append(accBeatPos)
            '''
            totDist.append(musicalDist/total)
            predTotDist.append(predMusicalDist/total)
            #totDist2a.append(musicalDist2a.item()/(len(dictDat["y"])/2))
            #totDist4a.append(musicalDist4a.item()/(len(dictDat["y"])/4))
            #totDist2b.append(musicalDist2b.item()/(len(dictDat["y"])/2))
            #totDist4b.append(musicalDist4b.item()/(len(dictDat["y"])/4))
            #totDist2c.append(musicalDist2b.item()/(len(dictDat["y"])/2))
            #totDist4c.append(musicalDist4b.item()/(len(dictDat["y"])/4))
            
        #Pinting time !
        #perp = perplexityCompute(sPrev, nby)

        print("rank for " + model + " on alphabet " + alph + ": " + str(np.mean(totRank)))
        print("perp for " + model + " on alphabet " + alph + ": " + str(np.mean(perp)))
        print("acc for " + model + " on alphabet " + alph + ": " + str(np.mean(totAcc)))
        #print("acc2 for " + model + " on alphabet " + alph + ": " + str(np.mean(totAcc2)))
        #print("acc4 for " + model + " on alphabet " + alph + ": " + str(np.mean(totAcc4)))
        print("accDownbeat for " + model + " on alphabet " + alph + ": " + str(np.average(totAccDownbeat,axis=0)))
        print("accPos for " + model + " on alphabet " + alph + ": " + str(np.average(totAccPos,axis=0)))
        print("accBeatPos for " + model + " on alphabet " + alph + ": " + str(np.average(totAccBeatPos, axis = 0)))
        print("Average Musical Distance for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist)))
        print("Average Prediction Musical Distance for " + model + " on alphabet " + alph + ": " + str(np.mean(predTotDist)))
        #print("Musical Distance2a for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist2a)))
        #print("Musical Distance4a for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist4a)))
        #print("Musical Distance2b for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist2b)))
        #print("Musical Distance4b for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist4b)))
        #print("Musical Distance2c for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist2c)))
        #print("Musical Distance4c for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist4c)))
        dictModel = {}
        dictModel["rank"] = np.mean(totRank)
        dictModel["perp"] = np.mean(perp)
        dictModel["acc"] = np.mean(totAcc)
        dictModel["accDownbeat"] = np.average(totAccDownbeat,axis=0)
        dictModel["accPos"] = np.average(totAccPos,axis=0)
        dictModel["accBeatPos"] = np.average(totAccBeatPos, axis = 0)
        dictModel["MusicalDist"] = np.mean(totDist)
        dictModel["PredMusicalDist"] = np.mean(predTotDist)
        #dictModel["MusicalDist2a"] = np.mean(totDist2a)
        #dictModel["MusicalDist4a"] = np.mean(totDist4a)
        #dictModel["MusicalDist2b"] = np.mean(totDist2b)
        #dictModel["MusicalDist4b"] = np.mean(totDist4b)
        #dictModel["MusicalDist2c"] = np.mean(totDist2c)
        #dictModel["MusicalDist4c"] = np.mean(totDist4c)
        dictFinalRes[model + "_" + alph] = dictModel
        
        
sauv = open(foldName + "_DictFinaltest.pkl","wb")
pickle.dump(dictFinalRes,sauv)
sauv.close()              
print("analyses completed")
            
#%%