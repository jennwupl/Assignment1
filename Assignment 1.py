#!/usr/bin/env python
# coding: utf-8

# # Assignment 1<br>
# <b>Jennifer Pei-Ling Wu<br>
# Jason Wei-Te Fang<br>
# Shirley Xinye Gong<br>
# Tracy Yu-Tung Huang</b><br>

# In[18]:


import pandas as pd
from pandasql import sqldf


# In[19]:


dataset = pd.read_csv("AB_test_data.csv")


# In[20]:


dataset


# In[21]:


A = dataset[dataset['Variant']=='A']
B = dataset[dataset['Variant']=='B']


# In[22]:


A


# In[23]:


q = '''select Variant, count(*)
from dataset
group by Variant'''

trials = sqldf(q)
trials


# In[24]:


q1 = '''select Variant, count(*)
from dataset
where purchase_TF = 1
group by Variant'''

successes = sqldf(q1)
successes


# # Part 1: Conduct AB Testing

# In[25]:


import math


# In[75]:


successes = []
sample_sets = []
i=1
p = 0.15206

while i < 11:    
    new_sample = B.sample(n=1158)
    
    sample_sets.append(new_sample.reset_index())
    
    new_successes = new_sample.purchase_TF.sum()
    
    p_hat = new_successes/1158
    
    z = (p_hat-p)/math.sqrt((p*(1-p)/1158))
    
    if z >= 1.64:
        successes.append(1)
    else: successes.append(0)
    i = i+1

print("Out of 10 tests, {} showed significant difference to support that Variant B performs better.".format(sum(successes)))


# <b> We can conclude that there is enough evidence to support that Variant B (with walkability assessment) performs better. </b>

# # Part 2: Optimal sample size

# In[80]:


alpha = 0.05
beta = 0.2
p0 = 0.15206
p1 = 0.1962
pbar = (p0 + p1) / 2
delta = p1 - p0

t_0025 = 1.96
t_02 = 0.842

n_optimal = ((t_0025 * math.sqrt(2*pbar*(1 - pbar)) + t_02*math.sqrt((p0)*(1-p0)+p1*(1-p1)))**2)/(delta**2)
n_optimal


# <b>We choose the optimal sample size to be 1158.</b>

# # Part 3: Sequential Testing

# In[70]:


import numpy as np


# In[71]:


f0_1 = 0.15206
f0_0 = 1-0.15206
f1_1 = 0.1962
f1_0 = 1-0.1962

A_bound = np.log(1/0.05)
B_bound = np.log(0.2)

result = []
length = []

for each_sample in sample_sets:
    i = 0
    recurrance = 0
    while i < 1158:
        if each_sample.purchase_TF[i] == True:
            recurrance = recurrance + np.log(f1_1 / f0_1)
        else:
            recurrance = recurrance + np.log(f1_0 / f0_0)
        
        if recurrance <= B_bound:
            length.append(i+1)
            result.append("Fail to reject H0, number of trials: {}".format(i+1))
            break
        elif recurrance >= A_bound:
            length.append(i+1)
            result.append("Reject H0, number of trials: {}".format(i+1))
            break
        else:
            i = i+1


# In[72]:


length


# In[73]:


sum(length)/len(length)


# In[74]:


result


# <b>We were able to stop all 10 tests prior to using the full sample. The average number of iterations required to stop the test is 306.5.</b>
