
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
from multiprocessing import  Pool
import pandas_profiling as pp
import re
import datetime
import os , glob , datetime
import pandas as pd
import numpy as np
from sklearn import decomposition, pipeline, metrics
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
from keras import optimizers
from collections import defaultdict
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


texts1 = list(df_train.d_text.values)
texts2 = list(df_test.d_text.values)
texts = texts1 #+ texts2


# In[ ]:


# Cleaning data - remove punctuation from every text
sentences = []
for ii in range(len(texts)):
    sentences = [re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', 
                        repl='', 
                        string=x
                       ).strip().split(' ') for x in texts[ii].split('\n') 
                      if not x.endswith('writes:')]
    sentences = [x for x in sentences if x != ['']]
    texts[ii] = sentences


# In[ ]:


all_sentences = []
for text in texts:
    all_sentences += text


# In[ ]:


for enu,text_list in enumerate(all_sentences):
    l = []
    for text in text_list:
        if text != "":
            l.append(text.lower())
    all_sentences[enu] = l


# ## Word embedding

# In[ ]:


from gensim.models import Word2Vec
model_wv = Word2Vec(all_sentences, min_count=3,size=200, workers=2, window=2, iter=10) 


# In[ ]:


from gensim.test.utils import get_tmpfile
word_vectors = model_wv.wv
word_vectors.save_word2vec_format('prealpha_word2vec_AG.txt', binary=False)


# In[ ]:


max_features = 20000
maxlen = 200


# In[ ]:


from keras import optimizers
from collections import defaultdict

EMBEDDING_FILE = 'prealpha_word2vec_AG.txt'  
embeddings_index = dict()
f = open(EMBEDDING_FILE, encoding="utf8")
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        pass
f.close()
print('Loaded %s word vectors.' % len(embeddings_index)) 


# ## Padding sequences

# In[ ]:


from keras.preprocessing import text, sequence
list_sentences_train = df_train["d_text"].fillna("CVxTz").values
y_dummy = pd.get_dummies(df_train['ag23'])
list_classes = y_dummy.columns
list_sentences_test = df_test["d_text"].fillna("CVxTz").values
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')
    


# In[ ]:


import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


embed_size =200
emb_mean,emb_std = 0.25, 0.25
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        try:
            embedding_matrix[i] = embedding_vector
        except:
            embedding_matrix[i] = embedding_vector[0:200]
adam = optimizers.RMSprop(lr=0.01)
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(y_dummy.shape[1], activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


batch_size = 128 # we can increase or decrease it, that will affect the time 
epochs = 40 # we can go for 100 epochs but time would be more 
file_path="weights_gs.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
callbacks_list = [checkpoint, early] #early
model.fit(X_t, y_dummy, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list, verbose=2)
model.load_weights(file_path)


# In[ ]:


y_test = model.predict(X_te)
df_test['pred']  = y_test.argmax(axis=1)
df_test['pred_prob'] = y_test[:,0]


# In[ ]:


df_test[df_test.ag23 == list_classes[df_test.pred]].shape[0] * 1.0/df_test.shape[0]

