# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 08:05:54 2016

@author: sem.f
"""

import pandas as pd
import re

review_col_name="review_text"

def sentenceSplitter(fp):
    #reads review file and splits review text into sentences

    df=pd.read_csv(fp)
    rev=df[review_col_name]
    out=[x for y in rev for x in re.split(';|\.|\!|\?',y) if x!=""]
    return out
    
def sentenceSplitterRev(fp):
    #reads review file and splits review text into sentences

    revList, sentList = [], []
    df=pd.read_csv(fp)
    rev=df[review_col_name]
    
    for i in rev:
        for j in re.split(';|\.|\!|\?',i):
            revList.append(i)
            sentList.append(j)

    return revList, sentList
    
def bootstrap(data, freq, idx):
                     
    freq = freq.set_index(idx)

    # This function will be applied on each group of instances of the same
    # class in `data`.
    def sampleClass(classgroup):
        cls = classgroup[idx].iloc[0]
        nDesired = freq.nostoextract[cls]
        nRows = len(classgroup)

        nSamples = min(nRows, nDesired)
        return classgroup.sample(nSamples)

    samples = data.groupby(idx).apply(sampleClass)

    # If you want a new index with ascending values
    # samples.index = range(len(samples))

    # If you want an index which is equal to the row in `data` where the sample
    # came from
    samples.index = samples.index.get_level_values(1)

    # If you don't change it then you'll have a multiindex with level 0
    # being the class and level 1 being the row in `data` where
    # the sample came from.

    return samples
 
def getClassSet(fp, filename):
    
    df=pd.read_csv(fp)
    sentence_label='Sentence'
    class_label='Class1'
    
    # define dictionaries
    class_dic={'Leakage': '__label__1', 
               'Absorbency': '__label__2',
               'Structure and Build Quality': '__label__3',
               'irritation/skin care/softness': '__label__4',
               'Fit and Comfort': '__label__5',
               'Usage': '__label__6',
               'Odor': '__label__7',
               'Price': '__label__8',
               'Appearance and Packaging': '__label__9',
               'Retailer': '__label__10',
               'Authenticity of the product': '__label__11',
               'Generic/Other': '__label__12',
               'Not Applicable': '__label__13'}
               
    class_freq = pd.DataFrame({'Class1':['Leakage', 
                                   'Absorbency', 
                                   'Structure and Build Quality',
                                   'irritation/skin care/softness',
                                   'Fit and Comfort',
                                   'Usage',
                                   'Odor',
                                   'Price',
                                   'Appearance and Packaging',
                                   'Retailer',
                                   'Authenticity of the product',
                                   'Generic/Other',
                                   'Not Applicable'],
                        'nostoextract':[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], })

    # get test set via bootstrap               
    test_set=bootstrap(df, class_freq, class_label)

    # get training set as difference between total set and bootstrapped test set
    df_concat = pd.concat([df, test_set])
    df_concat = df_concat.reset_index(drop=True)
    df_gpby = df_concat.groupby(list(df_concat.columns))
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
    training_set=df_concat.reindex(idx)
 
    # format sets for fasttext
    out_train=training_set.apply(lambda x: class_dic[x[class_label]] + ' , ' + x[class_label] + ' , ' + x[sentence_label], axis=1)              
    out_test=test_set.apply(lambda x: class_dic[x[class_label]] + ' , ' + x[class_label] + ' , ' + x[sentence_label], axis=1)
    
    # write files
    with open(filename + '_train', 'w') as f:
        for s in out_train:
            f.write(s + '\n')
        f.write('\n')
            
    with open(filename + '_test', 'w') as f:
        for s in out_test:
            f.write(s + '\n')
        f.write('\n')
        
        
        
def getSentimentSet(fp, filename):
    
    df=pd.read_csv(fp)
    sentence_label='Sentence'
    sentiment_label='Sentiment'
    
    # define dictionaries                
    sentiment_dict={'positive': '__label__1',
                    'neutral': '__label__2',
                    'negative': '__label__3'
                    }

    sentiment_freq = pd.DataFrame({'Sentiment':['positive', 
                                   'neutral', 
                                   'negative'],
                        'nostoextract':[100, 100, 100], })

    # get test set via bootstrap               
    test_set=bootstrap(df, sentiment_freq, sentiment_label)

    # get training set as difference between total set and bootstrapped test set
    df_concat = pd.concat([df, test_set])
    df_concat = df_concat.reset_index(drop=True)
    df_gpby = df_concat.groupby(list(df_concat.columns))
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
    training_set=df_concat.reindex(idx)
 
    # format sets for fasttext
    out_train=training_set.apply(lambda x: sentiment_dict[x[sentiment_label]] + ' , ' + x[sentiment_label] + ' , ' + x[sentence_label], axis=1)              
    out_test=test_set.apply(lambda x: sentiment_dict[x[sentiment_label]] + ' , ' + x[sentiment_label] + ' , ' + x[sentence_label], axis=1)
    
    # write files
    with open(filename + '_train', 'w') as f:
        for s in out_train:
            f.write(s + '\n')
        f.write('\n')
            
    with open(filename + '_test', 'w') as f:
        for s in out_test:
            f.write(s + '\n')
        f.write('\n')
        
def mergePrediction(test, pred):
    
    df_pred=pd.read_table(pred, sep=' ', header=None)
    df_test=pd.read_table(test, sep=' , ', header=None)
    df_combined=pd.concat([df_test,df_pred], axis=1)
    
    #df_combined.to_csv('merged_' + pred + '.csv')
    
    return df_combined

#df_combined.columns=['label','class','Sentence','pred1','prob1','pred2','prob2','pred3','prob3']
#source=pd.read_csv('data/pampers_set.csv')
#df_merged=pd.merge(df_combined, source, on='Sentence')
#df_merged.to_csv('pampers_out_merged.csv')
    
