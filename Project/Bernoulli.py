import re
import os
from collections import defaultdict
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
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
import pickle
from scipy.misc import comb
import glob
from operator import itemgetter 
import itertools
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report

def pickle_files(datastruct,filename,load,dump):
    if load:
        data = pickle.load(open(filename,'rb'))
        print 'end of loading file'
        return data
    if dump:
        pickle.dump(datastruct,open(filename,'wb'))
        print 'end of writing file'

def doc_length(X):
    doc_lengths = dict()
    for index,x in enumerate(X):
        doc_lengths[index] = sum(x)
    return doc_lengths

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
        data = open(f)
        temp = []
        for line in data:
            line = re.sub('\d+','',line)
            temp.append(re.findall(r'\w+|[!?#$]',line.lower()))
        
        temp = unfold_list(temp,[])        
        c = Counter(temp)
        tokens.append(c)

    return tokens     


def get_vocab_doc_freq(list_of_docs):
    print 'get_vocab_doc_freq'
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
    
    return vocab

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

def compute_alphai(X,indices,E=0.01):
    #     print 'compute_alphai'
    alpha = dict()
    for col in range(X.shape[1]):
        feat_col = X[:,col]
        a = 1.*(sum(feat_col[indices]) + E)/(len(indices)+ (2*E))
        #         print 'sum', (sum(feat_col[indices]))
        #         print 'n', len(indices)
        alpha[col] = a
    return alpha


def membership_fun_ber(x,class_val,alphai,priori):
    #     print 'membership_fun'
    s = 0
    for ind,i in enumerate(x):
        a = alphai[ind]
        #         print 'alphai', a
        #         print ''
        s += i*(math.log(a)) + (1-i)*(math.log(1-a))
    return (s+math.log(priori))   

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

def NB_Bern_exp(X,Y,x_predict):
    indices = dict()
    class_prior = dict()
    prediction = list()
    alphaj = dict()
    labels = np.unique(Y)
    doc_lengths = doc_length(X)
    
    for label in labels:
        indices[label] = indicator(Y,label)
        class_prior[label] = compute_prior(Y,label)
        alphaj[label] = compute_alphai(X,indices[label])
    
    for ind,x in enumerate(x_predict):
        temp = []
        for label in labels:
            temp.append(membership_fun_ber(x,label,alphaj[label],class_prior[label]))
        #             temp.append(membership_Bi(alphaj[label],class_prior[label],x))
        pred = labels[temp.index(max(temp))]
        prediction.append(pred)

    return prediction

def cross_validation_NBBer(spam_files,ham_files,k,verbose):
    
    #     spam_files = tokenize('dirname/spam/*',None)
    #     ham_files = tokenize('dirname/ham/*',None)
    
    files = spam_files + ham_files
    
    labels = [1]*len(spam_files) + [0]*len(ham_files)
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
        
        vocab,doc_freq = get_vocab_doc_freq(train_files)
        
        X_train = create_vector(train_files,vocab)
        Y_train = np.array(train_labels)
        X_test = create_vector(test_files,vocab)
        Y_test = np.array(test_labels)
        
        for l in classes:
            prior[l] = compute_prior(Y_train,l)
        
        Y_predict = NB_Bern_exp(X_train,Y_train,X_test)
        
        temp = accuracy_score(Y_test,Y_predict,normalize=True, sample_weight=None)
        c_matrix = confusion_matrix(Y_test, Y_predict)
        prec = precision_score(Y_test, Y_predict) 
        rec = recall_score(Y_test, Y_predict)  
        fm = f1_score(Y_test, Y_predict)
        
        accuracy.append(temp)
        recall.append(rec)
        precision.append(prec)
        f_measure.append(fm)
        
        fpr,tpr,thresholds = metrics.roc_curve(Y_test,Y_predict,pos_label=1)
        
        if verbose:
            print 'fold:', fold
            print 'False pos rate', fpr
            print 'True pos rate', tpr
            print 'accuracy:', temp
            print 'confusion_matrix'
            print c_matrix
            print 'prediction', prec
            print 'recall' , rec
            print (classification_report(Y_test,Y_predict,target_names=['ham','spam']))
            fold += 1

    avg_acc = sum(accuracy)/len(accuracy)
    avg_pre = sum(precision)/len(precision)
    avg_rec = sum(recall)/len(recall)
    avg_fm = sum(f_measure)/len(f_measure)
    
    print 'avg_accuracy: ' , avg_acc
    print 'avg_precision', avg_pre
    print 'avg_recall', avg_rec
    print 'avg_fmeasure', avg_fm


def main():
    spam_ = pickle_files(None,"new/spam_tokens_",load=True,dump=False)
    ham_ = pickle_files(None,"new/ham_tokens_",load=True,dump=False)
    cross_validation_NBBer(spam_,ham_,k=10,verbose=True)









