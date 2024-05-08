# -*- coding: utf-8 -*-
"""
Created on Wed May  8 02:09:17 2024

@author: Mohamed Elhakeem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.saving import register_keras_serializable
def tokenizer(text,max_len):
  #dic={'A':1,'G':1,'V':1,'I':2,'L':2,'F':2,'P':2,'Y':3,'M':3,'T':3,'S':3,'H':4,'N':4,'Q':4,'W':4,'K':5,'R':5,'D':6,'E':6,'C':7}
  dic={'A':1,'G':2,'V':3,'I':4,'L':5,'F':6,'P':7,'Y':8,'M':9,'T':10,'S':11,'H':12,'N':13,'Q':14,'W':15,'K':16,'R':17,'D':18,'E':19,'C':20}
  onehot=[]
  t=[]
  for i in range(len(text)):
    row=[]
    l=[]
    char=text[i].split(' ')
    for j in range(max_len):
      if j< len(char):
        row.append(dic[char[j]])
        r=np.zeros(20)
        r[dic[char[j]]-1]=1
      else:
        r=np.ones(20)*-1
        row.append(0)
      l.append(r)
    l=np.array(l)
    onehot.append(l)
    t.append(row)
  onehot=np.array(onehot)
  t=np.array(t)
  return t,onehot
def positional_encoding(positions, d):

    # initialize a matrix angle_rads of all the angles
    pos=np.arange(positions)[:, np.newaxis] #Column vector containing the position span [0,1,..., positions]
    k= np.arange(d)[np.newaxis, :]  #Row vector containing the dimension span [[0, 1, ..., d-1]]
    i = k//2
    angle_rads = pos/(10000**(2*i/d)) #Matrix of angles indexed by (pos,i)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    #adds batch axis
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

max_len=50
rnnamp_model = pd.read_csv('/data/combined.csv')
X=rnnamp_model['text']
y=np.array(rnnamp_model['labels'])
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)
X_train=np.array(X_train)
X_test=np.array(X_test)
X_train,onehot_train=tokenizer(X_train,max_len)
X_test,onehot_test=tokenizer(X_test,max_len)
@register_keras_serializable()
class TransformerModel(keras.Model):
    def __init__(self, input_vocab_size, d_model, num_heads, ff_dim, rate=0.1, maxlen=50):
        super(TransformerModel, self).__init__()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.PE = positional_encoding(maxlen, d_model)
        self.transformer_block = TransformerBlock_Encode(d_model, num_heads, ff_dim, rate)
        self.transformer_block2 = TransformerBlock_decode(d_model, num_heads, ff_dim, rate)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))
        self.fc3 = layers.Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))
        self.fc2 = layers.Dense(1, activation="sigmoid",kernel_regularizer=tf.keras.regularizers.l2(l2=0.1))

    def call(self, inputs, training):
        x = self.embedding(inputs)
        #x = x+self.PE
        y = self.transformer_block(x)
        y = self.dropout1(y)
        x = self.transformer_block2(x,y,y)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        return self.fc2(x)
class TransformerBlock_decode(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock_decode, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-1)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-1)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-1)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
    def call(self, inputs,q,k, training):
        attn_output = self.att(inputs, inputs,inputs)
        out1 = self.layernorm1(inputs + attn_output)
        attn_output1=self.att(q, k,out1)
        out2 = self.layernorm1(out1 + attn_output1)
        ffn_output = self.ffn(out2)
        return self.layernorm2(out2 + ffn_output)

# Define the TransformerBlock layer
class TransformerBlock_Encode(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock_Encode, self).__init__()
        self.con= layers.Conv1D(256,5,padding='same')
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-1)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-1)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training):
        inputs=self.con(inputs)
        attn_output = self.att(inputs, inputs,inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)



# Define your data loading and preprocessing here
# Example: X_train, y_train = load_data_and_preprocess()

# Define the model
input_vocab_size = 1024  # Replace with the actual vocabulary size
d_model = 256
num_heads = 2
ff_dim = 256

model = TransformerModel(input_vocab_size, d_model, num_heads, ff_dim)
initial_learning_rate = 0.0001
optimizer = Adam(learning_rate=initial_learning_rate)
# Compile the model
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])

callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),tf.keras.callbacks.ModelCheckpoint(filepath='AMAP1.h5', monitor='val_accuracy', save_best_only=True,mode='auto',save_weights_only=True)]

# Train the model
history = model.fit(X_train,y_train,epochs = 100,batch_size=32,validation_split=0.1,callbacks=[callback],verbose=1,shuffle= True)
# Plot training accuracy
plt.plot(history.history['accuracy'][:-10])
plt.plot(history.history['val_accuracy'][:-10])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training loss
plt.plot(history.history['loss'][:-10])
plt.plot(history.history['val_loss'][:-10])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
model.load_weights('AMAP1.h5')
model1=model
from sklearn import model_selection, metrics
y_pred=model.predict(X_train)
y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0

cv_preds = y_pred
print('combined train datasets')
name='deep learning'
print("%s: Accuracy %0.2f%%" % (name, 100*metrics.accuracy_score(y_train, cv_preds)))
print("%s: Precision-Recall %0.2f%%" % (name, 100*metrics.average_precision_score(y_train, cv_preds)))
print("%s: Matthews Coefficient %0.2f%%" % (name, 100*metrics.matthews_corrcoef(y_train, cv_preds)))
print("%s: Cohen Kappa Score %0.2f%%" % (name, 100*metrics.cohen_kappa_score(y_train, cv_preds)))
print("%s: ROC AUC Score %0.2f%%" % (name, 100*metrics.roc_auc_score(y_train, cv_preds)))
target_names = ['low 0', 'high 1']
print(metrics.classification_report(y_train, cv_preds, target_names=target_names))

# Predictions Validation Set
print('combined test datasets')
y_pred2=model.predict(X_test)
l=np.zeros(len(y_pred2))
l=l.reshape(-1,1)
l[y_pred2>=0.5]=1
l[y_pred2<0.5]=0
cv_preds2 = l
print("%s: Accuracy %0.2f%%" % (name, 100*metrics.accuracy_score(y_test, cv_preds2)))
print("%s: Precision %0.2f%%" % (name, 100*metrics.precision_score(y_test, cv_preds2)))
print("%s: Recall %0.2f%%" % (name, 100*metrics.recall_score(y_test, cv_preds2)))
print("%s: Matthews Coefficient %0.2f%%" % (name, 100*metrics.matthews_corrcoef(y_test, cv_preds2)))
print("%s: Cohen Kappa Score %0.2f%%" % (name, 100*metrics.cohen_kappa_score(y_test, cv_preds2)))
print("%s: ROC AUC Score %0.2f%%" % (name, 100*metrics.roc_auc_score(y_test, cv_preds2)))
target_names = ['low 0', 'high 1']
print(metrics.classification_report(y_test, cv_preds2, target_names=target_names))
rnnamp_model = pd.read_csv('/data/hlppredfuse.csv')
X=rnnamp_model['text']
y=np.array(rnnamp_model['labels'])
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)
X_train=np.array(X_train)
X_test=np.array(X_test)
X_train,onehot_train=tokenizer(X_train,max_len)
X_test,onehot_test=tokenizer(X_test,max_len)
input_vocab_size = 1024  # Replace with the actual vocabulary size
d_model = 256
num_heads = 2
ff_dim = 256
model2 = TransformerModel(input_vocab_size, d_model, num_heads, ff_dim)
initial_learning_rate = 0.0001
optimizer = Adam(learning_rate=initial_learning_rate)
# Compile the model
model2.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])

callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),tf.keras.callbacks.ModelCheckpoint(filepath='AMAP2.h5', monitor='val_accuracy', save_best_only=True,mode='auto',save_weights_only=True)]

# Train the model
history = model2.fit(X_train,y_train,epochs = 100,batch_size=32,validation_split=0.1,callbacks=[callback],verbose=1,shuffle= True)
model2.load_weights('AMAP2.h5')
model2.save_weights("AMAP2.h5")
from sklearn import model_selection, metrics
y_pred=model2.predict(X_train)
y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0

cv_preds = y_pred
print('hlppredfuse train datasets')
name='deep learning'
print("%s: Accuracy %0.2f%%" % (name, 100*metrics.accuracy_score(y_train, cv_preds)))
print("%s: Precision-Recall %0.2f%%" % (name, 100*metrics.average_precision_score(y_train, cv_preds)))
print("%s: Matthews Coefficient %0.2f%%" % (name, 100*metrics.matthews_corrcoef(y_train, cv_preds)))
print("%s: Cohen Kappa Score %0.2f%%" % (name, 100*metrics.cohen_kappa_score(y_train, cv_preds)))
print("%s: ROC AUC Score %0.2f%%" % (name, 100*metrics.roc_auc_score(y_train, cv_preds)))
target_names = ['low 0', 'high 1']
print(metrics.classification_report(y_train, cv_preds, target_names=target_names))

# Predictions Validation Set
print('hlppredfuse test datasets')
y_pred2=model2.predict(X_test)
l=np.zeros(len(y_pred2))
l=l.reshape(-1,1)
l[y_pred2>=0.5]=1
l[y_pred2<0.5]=0
cv_preds2 = l
print("%s: Accuracy %0.2f%%" % (name, 100*metrics.accuracy_score(y_test, cv_preds2)))
print("%s: Precision %0.2f%%" % (name, 100*metrics.precision_score(y_test, cv_preds2)))
print("%s: Recall %0.2f%%" % (name, 100*metrics.recall_score(y_test, cv_preds2)))
print("%s: Matthews Coefficient %0.2f%%" % (name, 100*metrics.matthews_corrcoef(y_test, cv_preds2)))
print("%s: Cohen Kappa Score %0.2f%%" % (name, 100*metrics.cohen_kappa_score(y_test, cv_preds2)))
print("%s: ROC AUC Score %0.2f%%" % (name, 100*metrics.roc_auc_score(y_test, cv_preds2)))
target_names = ['low 0', 'high 1']
print(metrics.classification_report(y_test, cv_preds2, target_names=target_names))
model2.save_weights("AMAP2.h5")

rnnamp_model = pd.read_csv('/data/rnnamp.csv')
X=rnnamp_model['text']
y=np.array(rnnamp_model['labels'])
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)
X_train=np.array(X_train)
X_test=np.array(X_test)
# Define the model
input_vocab_size = 1024  # Replace with the actual vocabulary size
d_model = 256
num_heads = 2
ff_dim = 256
model3 = TransformerModel(input_vocab_size, d_model, num_heads, ff_dim)
initial_learning_rate = 0.0001
optimizer = Adam(learning_rate=initial_learning_rate)
# Compile the model
model3.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])
callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),tf.keras.callbacks.ModelCheckpoint(filepath='AMAP3.h5', monitor='val_accuracy', save_best_only=True,mode='auto',save_weights_only=True)]

# Train the model
history = model3.fit(X_train,y_train,epochs = 100,batch_size=32,validation_split=0.1,callbacks=[callback],verbose=1,shuffle= True)
# Plot training accuracy
plt.plot(history.history['accuracy'][:-10])
plt.plot(history.history['val_accuracy'][:-10])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training loss
plt.plot(history.history['loss'][:-10])
plt.plot(history.history['val_loss'][:-10])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
from sklearn import model_selection, metrics
y_pred=model3.predict(X_train)
y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0

cv_preds = y_pred
print('rnnamp train datasets')
name='deep learning'
print("%s: Accuracy %0.2f%%" % (name, 100*metrics.accuracy_score(y_train, cv_preds)))
print("%s: Precision-Recall %0.2f%%" % (name, 100*metrics.average_precision_score(y_train, cv_preds)))
print("%s: Matthews Coefficient %0.2f%%" % (name, 100*metrics.matthews_corrcoef(y_train, cv_preds)))
print("%s: Cohen Kappa Score %0.2f%%" % (name, 100*metrics.cohen_kappa_score(y_train, cv_preds)))
print("%s: ROC AUC Score %0.2f%%" % (name, 100*metrics.roc_auc_score(y_train, cv_preds)))
target_names = ['low 0', 'high 1']
print(metrics.classification_report(y_train, cv_preds, target_names=target_names))

# Predictions Validation Set
print('rnnamp test datasets')
y_pred2=model3.predict(X_test)
l=np.zeros(len(y_pred2))
l=l.reshape(-1,1)
l[y_pred2>=0.5]=1
l[y_pred2<0.5]=0
cv_preds2 = l
print("%s: Accuracy %0.2f%%" % (name, 100*metrics.accuracy_score(y_test, cv_preds2)))
print("%s: Precision %0.2f%%" % (name, 100*metrics.precision_score(y_test, cv_preds2)))
print("%s: Recall %0.2f%%" % (name, 100*metrics.recall_score(y_test, cv_preds2)))
print("%s: Matthews Coefficient %0.2f%%" % (name, 100*metrics.matthews_corrcoef(y_test, cv_preds2)))
print("%s: Cohen Kappa Score %0.2f%%" % (name, 100*metrics.cohen_kappa_score(y_test, cv_preds2)))
print("%s: ROC AUC Score %0.2f%%" % (name, 100*metrics.roc_auc_score(y_test, cv_preds2)))
target_names = ['low 0', 'high 1']
print(metrics.classification_report(y_test, cv_preds2, target_names=target_names))
model3.save_weights("AMAP3.h5")
