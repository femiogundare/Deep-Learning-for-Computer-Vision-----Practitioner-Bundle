# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 01:53:06 2020

@author: femiogundare
"""


import numpy as np


def rank5_accuracy(preds, labels):
    rank1 = 0
    rank5 = 0
    
    for p, q in zip(preds, labels):
        p = np.argsort(p)[::-1]
        
        if q in p[:5]:
            rank5 +=1
            
        if q == p[0]:
            rank1 +=1
            
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))
    
    return rank1, rank5