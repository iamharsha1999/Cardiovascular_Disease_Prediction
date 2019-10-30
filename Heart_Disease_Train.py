import re, string
from nltk import sent_tokenize, word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.layers import LSTM,Dense, Embedding
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


def read_file(filepath):
    file = open(filepath, 'rt')
    text_data = file.read()
    file.close()

    return text_data

filename = '/home/harsha/Natural Language Processing/DataSet/Mark Twain- Yankee Connecticut.txt'
text_data = read_file(filename)



sentences = sent_tokenize(text_data)

##======= Splitting the Text Document Into Token ======##
tokenizer = Tokenizer(num_words=20000,oov_token= "oov", lower = False)
tokenizer.fit_on_texts(sentences)
vocab = tokenizer.word_index
print("No Of Words: " + str(len(vocab)))
word_freq = tokenizer.word_counts
nod = tokenizer.word_docs

##======= Dataset Preparation ==========##
input_sequences = []
for sent in sentences:
    token_list = tokenizer.texts_to_sequences([sent])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

max_length = max(len(x) for x in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_length, padding= 'pre'))

x,y = input_sequences[:,:-1], input_sequences[:,-1]
y = to_categorical(y, num_classes= len(vocab)+1)

##========= Model Definition ============##
model = Sequential()
model.add(Embedding(len(vocab)+1,10,input_length=max_length-1))
model.add(LSTM(32))
model.add(Dense(len(vocab)+1, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
plot_model(model, to_file='NLP_Model.png')

model.summary()
history = model.fit(x,y,epochs=100,verbose=1,batch_size=10000)

##Plotting the Training
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

