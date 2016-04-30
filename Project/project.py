import re
import os
from collections import defaultdict
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
#import numpy as np
from pylab import *
from decimal import Decimal
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from decimal import Decimal
from scipy.misc import comb
import glob
import codecs
import pickle
from operator import itemgetter 
import itertools
from sklearn import preprocessing

def unfold_list(temp,x):
    for t in temp:
        if isinstance(t,list):
            unfold_list(t,x)
        else:
            x.append(t)
    return x


def tokenize(dirname,n):
    tokens = []
    files = glob.glob(dirname)
    
    if n:
        files_list = files[:n]
    else:
        files_list = files
    for f in files_list:
        data = open(f,mode='r')
        temp = []
        for line in data:
            line = re.sub('\d+','',line)
            temp.append(re.findall(r'\w+|[!?#$]',line.lower()))
        
        temp = unfold_list(temp,[])        
        c = Counter(temp)
        for word in c.keys():
            total_word_count = sum(c.values())
            c[word] /= 1.*total_word_count
        tokens.append(c)
    
    return tokens        

def pickle_files(datastruct,filename,load,dump):
    
    if load:
        data = pickle.load(open(filename,mode='r'))
        print ('end of loading file')
        return data
    if dump:
        pickle.dump(datastruct,open(filename,'wb'))
        print ('end of writing file')

def get_vocab_doc_freq(list_of_docs):
    tokens = []
    
    for doc in list_of_docs:
        for token in doc.keys():
            if token not in tokens:
                tokens.append(token)
    
            
    doc_freq = defaultdict(int)
    for i,t in enumerate(tokens):
        for doc in list_of_docs:
            if t in doc.keys():
                doc_freq[t] += 1
     
    vocab = [feature for feature in doc_freq if doc_freq[feature] >= 5]
    
    if len(vocab) > 3000:
        return vocab[:3000]
    else:
        return vocab
            

def create_vector(files,vocab):
    X = []
    for f in files:
        temp = [0]*len(vocab)
        for i,feature in enumerate(vocab):
            if feature in f.keys():
                temp[i] = f[feature]
        X.append(temp)
    
    X = np.array(X)
    return X

def indicator(Y,class_val):
    l = list()
    for idx,y in enumerate(Y):
        if y == class_val:
            l.append(idx)
    return l

def compute_prior(Y,class_val):
    m = len(Y)
    indices = indicator(Y,class_val)
    return (1.*len(indices)/m)

def calc_mean(X,Y,label):
    indices = indicator(Y,label)
    x = X[indices]
    return np.mean(x,axis=0)

def calc_std_dev(X,Y,label):
    indices = indicator(Y,label)
    x = X[indices]
    return np.std(x,axis=0)

def likelihood_function(x,mean,sigma,vocab):
#    print ('likelihood_function')
    p = 0.0
    for i,xi in enumerate(x):
        if xi != 0.0 or sigma[i] != 0.0:
#        if sigma[i] > math.pow(10,-3):
#        print (xi)
#        print (sigma[i])
#        print (mean[i])
            c = (1/(sigma[i]*math.sqrt(2*math.pi)))
#        print (c)
            a = ((xi-mean[i])**2)/(2*(sigma[i]**2))
#        print (a)
#        print (math.exp(-a))
            b = c*math.exp(-a)
#        print ("b")
#        print (b)
#        print("")
            if b>0.0:
                p += math.log(b)
    return p


def predict_y(X,Y,labels,x_predict,prior,vocab):
    print ('predict_y')
    mean = {}
    sigma = {}
    y = []
   
    for l in labels:
        mean[l] = calc_mean(X,Y,l)
        sigma[l] = calc_std_dev(X,Y,l)
    

    for ind,x in enumerate(x_predict):
        temp = []
        for l in labels:
            a = (likelihood_function(x,mean[l],sigma[l],vocab))
#            print (a)
#            print (math.log(prior[l]))
            p = (a) + math.log(prior[l])
            temp.append(p)
#        print (temp)
        prob = 1.*temp[1]/sum(temp)
        if prob > 0.5:
            y.append(1)
        else:
            y.append(0)

    print (x_predict.shape)
    print (len(y))
    return y


def cross_validation(dirname,k,verbose):

#    spam = pickle_files(None,'spam_tokens',load=True,dump=False)
#    ham = pickle_files(None,'ham_tokens',load=True,dump=False)
#    voc = pickle_files(None,'vocabulary',load=True,dump=False)
#    files = spam + ham
#    labels = [1]*len(spam) + [0]*len(ham)
#    voc = pickle_files(None,'vocabulary',load=True,dump=False)

    spam = tokenize(dirname+"spam/*",None)
    ham = tokenize(dirname+"ham/*",None)
    files = spam+ham
    labels = [1]*len(spam) + [0]*len(ham)

    classes = [0,1]
    
    
    accuracy = list()
    precision = list()
    recall = list()
    f_measure = list()
    prior = dict()
    fold = 1
    
    for train_ind, test_ind in KFold(len(files),k,shuffle=True,random_state=5):
         
        train_files = itemgetter(*train_ind)(files)
        test_files = itemgetter(*test_ind)(files)
        train_labels = itemgetter(*train_ind)(labels)
        test_labels = itemgetter(*test_ind)(labels)
        
        vocab = get_vocab_doc_freq(train_files)
        
        X_train = create_vector(train_files,voc)
        Y_train = np.array(train_labels)
        X_test = create_vector(test_files,voc)
        Y_test = np.array(test_labels)
        
        for l in classes:
            prior[l] = compute_prior(Y_train,l)
            
       
        Y_predict = predict_y(X_train,Y_train,classes, X_test,prior,voc)
        
        
        acc = accuracy_score(Y_test,Y_predict,normalize=True, sample_weight=None)
        c_matrix = confusion_matrix(Y_test, Y_predict)
        prec = precision_score(Y_test, Y_predict) 
        rec = recall_score(Y_test, Y_predict)  
        fm = f1_score(Y_test, Y_predict)
    
        accuracy.append(acc)
        recall.append(rec)
        precision.append(prec)
        f_measure.append(fm)
        
        if verbose:
            print ('fold:', fold)
            print ('accuracy:', acc)
            print ('confusion_matrix')
            print (c_matrix)
            print ('prediction', prec)
            print ('recall' , rec)
            print ('f-measure',fm)
            print ('')
        fold += 1
    
    print (accuracy)
    print (precision)
    print (recall)
    print (f_measure)
    print (' ')
    
    avg_acc = sum(accuracy)/len(accuracy)
    avg_pre = sum(precision)/len(precision)
    avg_rec = sum(recall)/len(recall)
    avg_fm = sum(f_measure)/len(f_measure)
   
    print ('avg_accuracy: ' , avg_acc)
    print ('avg_precision', avg_pre)
    print ('avg_recall', avg_rec)
    print ('avg_fmeasure', avg_fm)

def save_likelihood(filename):
    print ('saving likelihood')
    spam = pickle_files(None,'spam_tokens',load=True,dump=False)
    ham = pickle_files(None,'ham_tokens',load=True,dump=False)
    voc = pickle_files(None,'vocabulary',load=True,dump=False)
#    voc = vocabulary[:3000]

#    print (voc)
    files = spam + ham
    X = create_vector(files,voc)
    Y = [0]*len(ham) + [1]*len(spam)

    print ('X', X.shape)
    labels = [0,1]
    mean = dict()
    sigma = dict()
    likelihood = []
    
    for l in labels:
        mean[l] = calc_mean(X,Y,l)
        sigma[l] = calc_std_dev(X,Y,l)
    
    for i,x in enumerate(X):
        print ("vector "+ str(i))
        print ("")
        if (i%50 == 0):
            likelihood.append((likelihood_function(x,mean[0],sigma[0],voc),likelihood_function(x,mean[1],sigma[1],voc)))
            pickle_files(likelihood,(filename+"_"+str(i)),load=False,dump=True)
            likelihood = []
        else:
            likelihood.append((likelihood_function(x,mean[0],sigma[0],voc),likelihood_function(x,mean[1],sigma[1],voc)))
        
#    pickle_files(likelihood,filename,load=False,dump=True)
    print ('end of pickling likelihood values')


def main():
#    save_likelihood("likelihood")
    cross_validation("data/enron2/",k=10,verbose=True)
    return

main()
      
