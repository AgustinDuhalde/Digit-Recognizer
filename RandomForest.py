
# coding: utf-8

# In[1]:

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy.sparse as sc
import pandas as pd
df=pd.read_csv('trainOrig.csv', sep=',',header=None)
#test =pd.read_csv('test.csv', sep=',',header=None)

cont_pos = 0 
cont_neg = 0
df2= df.ix[1:].as_matrix()
df2 = df2.astype(np.float)

#test = test.as_matrix().astype(np.float)
#test_res = [x[0] for x in test]
#test_pixel = [x[1:785] for x in test]
resultados =[x[0] for x in df2[:30000]]
test_res = [x[0] for x in df2[30001:]]






rf = RandomForestClassifier(n_estimators=100)
rf.fit([x[1:785] for x in df2[:30000]],resultados)

res_predicc = rf.predict([x[1:785] for x in df2[30001:]])
for x in range (len(res_predicc)):
    if res_predicc[x] == test_res[x]:
        cont_pos+=1
    else :
        cont_neg+=1




#df_csr = sc.csr_matrix(df2)


# In[54]:

img = df2[1][1:]
num = df2[1][0]
nrows,ncols = 28,28
grid = img.reshape((nrows, ncols))
plt.matshow(grid) 
plt.show()
print num 


# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt


#create the training & test sets, skipping the header row with [1:]
dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:400]    
target = [x[0] for x in dataset]
print 'leido target y cargado el train'
train = [x[1:785] for x in dataset]
print 'leido'
test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:400]
print  '3'
#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100)
print '4'
    
rf.fit(train, target)
print rf.predict(test)
    #savetxt('submission2.csv', rf.predict(test), delimiter=',', fmt='%f')

