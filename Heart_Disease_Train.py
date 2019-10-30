##Importing All the Required Packages
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Creating the Model Architecture
def nn_model():
    model = Sequential()
    ## ReLu Block 1
    model.add(Dense(32,   input_dim = 11))
    model.add(BatchNormalization())

    ##ReLu Block 2
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())

    ##ReLu Block 3
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())

    ##Sigmoid Output Layer
    model.add(Dense(1, activation='sigmoid'))

    return model

## Filepath of the CSV File
csv_file_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/cardio_train.csv'
df = pd.read_csv(csv_file_path, delimiter=';')
df = df.dropna()
df = df.iloc[:,1:13]
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
batch_size = 1024

##Compiling the Model
model = nn_model()
model.compile(loss = 'binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train, epochs= epochs, batch_size= batch_size, verbose = 1, callbacks= callbacks, validation_data=(x_val, y_val))


## Taking the input from the user
# test = []
# test.append(int(input('Enter the Age:')) * 365)
# test.append(int(input('Enter the Gender (1:Male 2:Female) : ')))
# test.append(int(input('Enter the Height in cms: ')))
# test.append(int(input('Enter the Weight in Kgs: ')))
# test.append(150)
# test.append(80)
# test.append(int(input('Enter the Cholesterol (1:Normal 2:Above Normal 3:Well Above Normal)')))
# test.append(int(input('Enter Glucose (1:Normal 2:Above Normal 3:Well Above Normal)')))
# test.append(int(input('Enter the Smoking Habit (0: Smoke 1:Doesnt Smoke)')))
# test.append(int(input('Enter the Alcoholic Habit (0:Alcholic 1:Not Alcholoic)')))
# test.append(int(input('Enter the Physical Acitivity (0:Physically Active 1:Physically Not Active')))
# df.append(test)
# df =  scaler.fit_transform(df)
#

##Predictiom
# weights_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/Weights/weights-improvement-82-0.73.hdf5'
# model.load_weights(weights_path)
# result = model.predict(np.expand_dims(df[-1], axis =0))
# print(df[-1])
# print('The Probablity is :{}'.format(result))
# if result>0.7:
#     print('Has a cardiovascular Disease')
# else:
#     print('Doesnt Have a Cardiovascular Disease')
