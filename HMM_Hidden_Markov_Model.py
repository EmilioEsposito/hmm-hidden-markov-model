# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:45:39 2016

@author: Emilio Esposito
All probabilities calculated use log-probabilities, EXCEPT for the inital dfs: emit, trans, prior
"""

import pandas as pd
import numpy as np
from logsum import log_sum_lst

# INPUTS
emit = pd.read_csv("emit.csv",index_col=0)
trans = pd.read_csv("trans.csv",index_col=0)
prior = pd.read_csv("prior.csv",index_col=0)
data = pd.read_csv("obs.csv")
data["comb"] = data.iloc[:,2:6].apply(lambda x: ''.join(x.map(str)), axis=1)

# get obs for user1
obs = data[data.user==1]["comb"]

states = emit.index

print "original trans", trans

# HMM ALGORITHMS

# FORWARD
def forward(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))

    for t in trellis.columns[:-1]:
        if t == 0:
            trellis.loc[:,t] = np.log(prior.loc[:,"prob"]) + np.log(emit.loc[:, obs[t]])
        
        for i_state in states:
            
            log_summation = log_sum_lst([trellis.loc[j_state,t] + np.log(trans.loc[j_state,i_state]) for j_state in states])
            
            trellis.loc[i_state, t+1] = np.log(emit.loc[i_state, obs[t+1]]) + log_summation
    
    # get total prob
    # sum the last column across all states
    total_prob = log_sum_lst(trellis.loc[:,trellis.columns[-1:]].values)
    print "fwd", total_prob[0]
    
    return trellis, total_prob
      
# BACKWARD
def backward(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))
    for t in reversed(trellis.columns):
        if t == trellis.columns[-1]:
            trellis.loc[:,t] = np.log(1)
        else:
            for i_state in states:
                
                log_summation = log_sum_lst([trellis.loc[j_state,t+1] + np.log(trans.loc[i_state,j_state]) + np.log(emit.loc[j_state, obs[t+1]]) for j_state in states])
                
                trellis.loc[i_state, t] = log_summation
    
    # get total prob
    total_prob = log_sum_lst([np.log(prior.loc[i_state,"prob"]) + np.log(emit.loc[i_state, obs[0]]) + trellis.loc[i_state, 0] for i_state in states])
    print "bkwd", total_prob
    
    return trellis

        
# VITERBI
def viterbi(prior, trans, emit, states, obs):
    # create trellis like trellis[t][state] e.g. trellis[0]["s1"]
    trellis = pd.DataFrame(np.zeros([len(states),len(obs)]), index=states, columns=range(len(obs)))
    
    path = pd.DataFrame(index = states, columns=range(len(obs)))
       
    for t in trellis.columns[:-1]:
    
        if t == 0:
            trellis.loc[:,t] = np.log(prior.loc[:,"prob"]) + np.log(emit.loc[:, obs[t]])
            path.loc[:, t] = states
        
        for i_state in states:
            
            max_prob = np.max([trellis.loc[j_state,t] + np.log(trans.loc[j_state,i_state]) for j_state in states])
            
            trellis.loc[i_state, t+1] = np.log(emit.loc[i_state, obs[t+1]]) + max_prob
            path.loc[i_state, t+1] = trellis.loc[:,t+1].idxmax()
    
    max_state = trellis.iloc[:,-1].idxmax()
            
    return path.loc[max_state]

# FORWARD-BACKWARD Baum Welch algorithm
def fwd_bwd():
    # get forward trellis and total prob of trellis
    f_trellis, total_prob = forward(prior, trans, emit, states, obs)
    
    # get backward trellis
    b_trellis = backward(prior, trans, emit, states, obs)
    
    # calculate Xi (aka ksi)
    # Probability of transiting from to Si to Sj at time t given output O
    def ksi(t, i_state, j_state):
        # calc ksi_t_ij using log prob
        # Probability of transiting from to Si to Sj at time t given output O
        ksi_t_ij = f_trellis.loc[i_state,t] + np.log(trans.loc[i_state, j_state]) + np.log(emit.loc[j_state, obs[t+1]]) + b_trellis.loc[j_state, t+1] - total_prob
        return ksi_t_ij[0]
    
    # update trans matrix A
    for i_state in states:
        # initialize numerators and denominator
        aij_numerator_lst = []
        aij_denominator = np.log(0)
        
        # calc numerators, and accumulate numerators into a denominator
        for j_state in states:
            # calc numerator for j_state
            aij_numerator = log_sum_lst([ksi(t, i_state, j_state) for t in list(obs.index)[:-1]])
            # add it to a list, we'll use it next loop
            aij_numerator_lst.append(aij_numerator)
            # the running total of numerators becomes the denominator 
            aij_denominator = log_sum_lst([aij_denominator, aij_numerator])
        # update trans 
        for j_state, aij_numerator in zip(states, aij_numerator_lst):
            
            # update trans probs and convert from log prob to normal prob by using exponentiation
            trans.loc[i_state, j_state] = np.exp(aij_numerator - aij_denominator)
    
    # update emissions emit matrix B
    for j_state in states:
        # initialize numerators and denominator
        bij_numerator_lst = []
        bij_denominator = np.log(0)
        
        # calculate denominator        
        ksi_t_ij_lst = [log_sum_lst([ksi(t, i_state, j_state) for i_state in states]) for t in list(obs.index)[:-1]]
        bij_denominator = log_sum_lst(ksi_t_ij_lst)
        
        # loop through all k emission output possibilities
        for k in emit.columns:
            # calculate numerator
            bij_numerator_lst = [np.log(0)]
            
            # calculate ksi where k == observed output and store it in a list
            for t in range(len(obs)-1):
                if obs.loc[t+1] == k:
                    for i_state in states:
                        bij_numerator_lst.append(ksi(t, i_state, j_state))
                        
            # sum the list to get the numerator value
            bij_numerator = log_sum_lst(bij_numerator_lst)
            
            # update emission probs and convert from log prob to normal prob by using exponentiation
            emit.loc[j_state, k] = np.exp(bij_numerator - bij_denominator)
            
    # return total_prob so that we can test for convergance between consecutive runs
    return total_prob

# initialize probs far apart
last_total_prob = np.log(0)
total_prob = np.log(1)

# TRAINING
# run fwd-bwd algo until the probabilies converge
i = 0
while np.abs(last_total_prob - total_prob) > .1:
    i+=1
    print "Iteration #", i, " of fwd-bwd algo..."
    last_total_prob = total_prob
    total_prob = fwd_bwd()

# DECODING
viterbi_path = viterbi(prior, trans, emit, states, obs)

# save the final outputs

trans_final_out = open("trans_final.csv","w")
trans_final_out.write(trans.to_csv())
trans_final_out.close()

emit_final_out = open("emit_final.csv","w")
emit_final_out.write(emit.to_csv())
emit_final_out.close()

viterbi_path_final_out = open("viterbi_path_final.csv","w")
viterbi_path_final_out.write(pd.DataFrame(viterbi_path).to_csv())
viterbi_path_final_out.close()
