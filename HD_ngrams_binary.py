#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:13:43 2019

@author: denkle
"""
import numpy as np

def ngram_encode_bin(str, HD_aphabet, aphabet, n_size): # method for mapping n-gram statistics of a word to an N-dimensional HD vector
    HD_ngram = np.zeros(HD_aphabet.shape[1]) # will store n-gram statistics mapped to HD vector
    full_str = '#' + str + '#' # include extra symbols to the string
    
    #adjust the string for n-gram size
    
    if n_size == 1:
        full_str_e=full_str            
    else:
        full_str_e=full_str[:-(n_size-1)]    
        
    for il, l in enumerate(full_str_e): # loops through all n-grams
        hdgram = HD_aphabet[aphabet.find(full_str[il]), :] # picks HD vector for the first symbol in the current n-gram
        
        for ng in range(1, n_size): #loops through the rest of symbols in the current n-gram
                hdgram = np.logical_xor( hdgram, np.roll(HD_aphabet[aphabet.find(full_str[il+ng]), :], ng)) # two operations simultaneously; binding via XOR; rotation via cyclic shift

        HD_ngram += hdgram # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram

    HD_ngram_norm =HD_ngram/len(full_str_e)  # normalizes HD-vector by number of n-grams. So value in each position is between 0 and 1    
    return HD_ngram_norm # output normalized HD mapping

N = 1000 # set the desired dimensionality of HD vectors
n_size=2 # n-gram size
aphabet = 'abcdefghijklmnopqrstuvwxyz#' #fix the alphabet. Note, we assume that capital letters are not in use 
np.random.seed(1) # for reproducibility
HD_aphabet = (np.random.randn(len(aphabet), N) < 0)  # generates binary {0, 1}^N HD vectors; one random HD vector per symbol in the alphabet

str='jump' # example string to represent using n-grams
HD_ngram = ngram_encode_bin(str, HD_aphabet, aphabet, n_size) # HD_ngram is a projection of n-gram statistics for str to N-dimensional space. It can be used to learn the word embedding