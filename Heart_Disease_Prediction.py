##Importing All the Required Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy

## Creating the Model Architecture
def nn_model():
    model = Sequential()

    model.add(Dense(32,   input_dim = 11))

    # model.add(Dense(16, activation='relu'))

    model.add(Dense(8, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    return model

## Filepath of the CSV File
csv_file_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/cardio_train.csv'
df = pd.read_csv(csv_file_path, delimiter=';')
df = df.dropna()
print(df.head())
print('Total No Of Rows: {}'.format(df.shape[0]))
print('Total No of Columns: {}'.format(df.shape[1]))

## Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
df =  scaler.fit_transform(df)
##Plotting the Correlation Matrix
# corr = df.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = numpy.arange(0,13,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# plt.show()

## Splitting the Dataset
x_train,x_val,y_train,y_val = train_test_split(df[:,1:12],df[:,-1], test_size= 0.3, random_state=32)


##Creating Checkpoints
weights = '/home/harsha/Machine_Learning_Project/AI_J_Component/Weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoints = ModelCheckpoint(weights, verbose= 1, monitor= 'val_acc', save_best_only= True, mode = 'max')
callbacks = [checkpoints]

epochs = 100
batch_size = 512

##Compiling the Model
model = nn_model()
model.compile(loss = 'binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
model.summary()
# model.fit(x_train,y_train, epochs= epochs, batch_size= batch_size, verbose = 1, callbacks= callbacks, validation_data=(x_val, y_val))


##Predictiom
weights_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/Weights/weights-improvement-82-0.73.hdf5'
model.load_weights(weights_path)
result = model.predict(np.expand_dims(df[47844, 1:12], axis=0))
if result>0.5:
    print(1)
else:
    print(0)


