#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:57:22 2020

https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296

@author: mahathir
"""
#%%
%reset -f
import nltk
#nltk.download()
#%%
from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing
#%%
sentences = brown.sents()
#%%
#%%
EMB_DIM = 300
w2v = Word2Vec(sentences,size=EMB_DIM, window=5, min_count=5, negative=15,iter=10,workers=multiprocessing.cpu_count())
#%%
word_vectors = w2v.wv
result = word_vectors.similar_by_word("Saturday")
print("Most similar to 'Saturday':\n",result[:3])

result = word_vectors.similar_by_word("money")
print("Most similar to 'money':\n",result[:3])

result = word_vectors.similar_by_word("child")
print("Most similar to 'child':\n",result[:3])

result = word_vectors.most_similar(positive=['child'],negative=['person'])
print("Most similar to 'child' but dissimilar to 'person':\n",result[:3])

#%%
from nltk.corpus import conll2000
from gensim.models import Word2Vec
import tensorflow as tf
from keras.layers import Dense, Embedding, Activation, Flatten
from keras import Sequential
from keras.utils import to_categorical
import numpy as np
import collections

#%%
train_words = conll2000.tagged_words("train.txt")
test_words = conll2000.tagged_words("test.txt")
print(train_words[:10])
#%%
def get_tag_vocabulary(tagged_words):
    tag2id={}
    for item in tagged_words:
        tag  = item[1]
        tag2id.setdefault(tag,len(tag2id))
    return tag2id

word2id = {k: v.index for k,v in word_vectors.vocab.items()}
tag2id = get_tag_vocabulary(train_words)
#%%
def get_int_data(tagged_words, word2id, tag2id):
    X,Y = [],[]
    unk_count = 0
    
    for word, tag in tagged_words:
        Y.append(tag2id.get(tag))
        if word in word2id:
            X.append(word2id.get(word))
        else:
            unk_count += 1
    print("Data created. Percentage of unkown words: %.3f"%(unk_count/len(tagged_words)))
    return np.array(X),np.array(Y)

X_train, Y_train = get_int_data(train_words, word2id, tag2id)
X_test, Y_test = get_int_data(test_words, word2id, tag2id)

Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)
#%%
def add_new_word(new_word, new_vector, new_index, embedding_matrix, word2id):
    embedding_matrix = np.insert(embedding_matrix,[new_index],[new_vector],axis = 0)
    word2id = {word: (index+1) if index >= new_index else index
               for word,index in word2id.items()}
    word2id[new_word] = new_index
    return embedding_matrix, word2id

UNK_INDEX = 0
UNK_TOKEN = "UNK"

embedding_matrix = word_vectors.vectors
unk_vector = embedding_matrix.mean(0)
embedding_matrix, word2id = add_new_word(UNK_TOKEN,unk_vector,UNK_INDEX, embedding_matrix, word2id)
#%%
def get_int_data(tagged_words, word2id, tag2id):
    X,Y = [],[]
    unk_count = 0
    for word,tag in tagged_words:
        Y.append(tag2id.get(tag))
        if word in word2id:
            X.append(word2id.get(word))
        else:
            X.append(UNK_INDEX)
            unk_count += 1
    
    print("Data created. Percentage of unknown words: %.3f " % (unk_count/len(tagged_words)))
    return np.array(X),np.array(Y)

X_train,Y_train = get_int_data(train_words, word2id, tag2id)
X_test, Y_test = get_int_data(test_words, word2id, tag2id)
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)
#%%
HIDDEN_SIZE = 50
BATCH_SIZE = 128

def define_model(embedding_matrix, class_count):
    vocab_length = len(embedding_matrix)
    model = Sequential()
    
    model.add(Embedding(input_dim=vocab_length,
                        output_dim = EMB_DIM,
                        weights = [embedding_matrix],
                        input_length=1))
    model.add(Flatten())
    model.add(Dense(HIDDEN_SIZE))
    model.add(Activation("tanh"))
    model.add(Dense(class_count))
    model.add(Activation("softmax"))
    
    model.compile(optimizer = tf.optimizers.Adam(),loss = "categorical_crossentropy", metrics=["accuracy"])
    return model

pos_model = define_model(embedding_matrix,len(tag2id))
pos_model.summary()

pos_model.fit(X_train,Y_train,batch_size = BATCH_SIZE,epochs = 1,verbose=1)
#%%
def evaluate_model(model,id2word,x_test,y_test):
    _,acc = model.evaluate(x_test,y_test)
    print("Accuracy: %.2f " % acc)
    y_pred = model.predict_classes(x_test)
    error_counter = collections.Counter()
    
    for i in range(len(x_test)):
        correct_tag_id = np.argmax(y_test[i])
        if y_pred[i] != correct_tag_id:
            word = id2word[x_test[i]]
            error_counter[word]+=1
    
    print("Most common errors: \n",error_counter.most_common(10))
    
id2word = sorted(word2id, key= word2id.get)
evaluate_model(pos_model,id2word,X_test,Y_test)
#%%