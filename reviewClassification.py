

from keras.datasets import imdb

"""## Data Preperation"""

((XT,YT) , (Xt,Yt)) = imdb.load_data(num_words = 10000)

import numpy as np
print(len(XT))
print(len(Xt))
print(XT[0])

"""To check how the data looks like we use get_word_index() which returns word index dictionary"""

word_idx = imdb.get_word_index()
print(word_idx.items())

idx_word = dict([val,key] for (key,val) in word_idx.items())
print(idx_word.items())

actual_review = ' '.join([idx_word.get(idx-3,'?') for idx in XT[0]])
actual_review

# Next Step -  Vectorize the Data
# Vocab size - 10,000 we will make every sentence is represented by vector of len 10k [0010100..1....1..0..1..]

def vectorize_sentences(sentences,dim = 10000):
  outputs = np.zeros((len(sentences),10000))

  for i,idx in enumerate(sentences):
    
    outputs[i,idx] = 1

  return outputs

X_train = vectorize_sentences(XT)
X_test = vectorize_sentences(Xt)
print(X_train.shape)
print(X_train[0].shape)
X_train[0]

print(YT.shape)

#converting YT,Yt to numpy array
Y_train = np.asarray(YT).astype('float32')
Y_test = np.asarray(Yt).astype('float32')
print(Y_train.shape)

"""###  Model Architecture
-  Use fully connected/Dense Layers with ReLu Activation
- 2 Hidden Layers with 16 Units each
- 1 Output layer with 1 unit(Sigmoid function)
"""

from keras import models
from keras.layers import Dense

#define the model
model = models.Sequential()
model.add( Dense ( 16,activation = 'relu',input_shape = (10000,) ) )
model.add(Dense(16,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop' , loss = 'binary_crossentropy',metrics =['accuracy'])

model.summary()

# Training and Validation
x_val = X_train[:5000]
x_train_new = X_train[5000:]

y_val = Y_train[:5000]
y_train_new = Y_train[5000:]

hist = model.fit(x_train_new , y_train_new , epochs = 15, batch_size = 512 , validation_data=(x_val,y_val) )

# Visualise results

import matplotlib.pyplot as plt

h = hist.history

plt.plot(h['val_loss'] , label = "Validation loss")
plt.plot(h['loss'] , label = "Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(h['val_accuracy'] , label = "Validation acc")
plt.plot(h['accuracy'] , label = "Training acc")
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()
plt.show()

model.evaluate(X_train ,Y_train)[1]

model.evaluate(X_test,Y_test)[1]

model.predict(X_test)

