#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
numpy.random.seed(7)


# In[2]:


#downloading only 5000 top words from the dataset, these will be considered as total number of words in our vocabulary
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)


# In[5]:


#padding input sequences for every review
#this is done so that we will be able to input words in batches to our LSTM
#Eg:- x1=Review 1, x2= Review 2 then x11,x12,x13,x14... are words in review 1 and x21,x22,x23,x24,x25.. are words in review 2 and so on.
#At every LSTM , we input in 'batches' i.e. (x11,x21,x31,x41,x51) given at one time (when batch size =5)
#hence we need padding to make all the review sentences of same length
max_review_length = 600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(X_train.shape)
print(X_train[1])


# In[11]:


#LSTM MODEL IN KERAS
embedding_vector_length = 32 #embedding vector with 32 activations
model = Sequential()
model.add(Embedding(top_words+1, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))#100 LSTM units taken at same time 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[14]:


#fit the model
model.fit(X_train, y_train, nb_epoch=2, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


#within two epochs ,LSTM model is able to give us accuracy of more than 85 percent

