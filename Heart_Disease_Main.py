from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def nn_model():
    model = Sequential()

    model.add(Dense(32,   input_dim = 11))
    model.add(BatchNormalization())

    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    return model

## Filepath of the CSV File
csv_file_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/cardio_train.csv'
df = pd.read_csv(csv_file_path, delimiter=';')
df = df.dropna()
df = df.iloc[:,1:12]
print(df.head())
print('Total No Of Rows: {}'.format(df.shape[0]))
print('Total No of Columns: {}'.format(df.shape[1]))

## Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))

##Compiling the Model
model = nn_model()
model.compile(loss = 'binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
model.summary()

## Taking the input from the user
age = int(input('Enter the Age:')) * 365
gender= int(input('Enter the Gender (1:Male 2:Female) : ')) # 0 for women, 1 for men
height= int(input('Enter the Height in cms: ')) # in cm
weight= int(input('Enter the Weight in Kgs: ')) # in kilograms
systolicbloodpressure= 120 # Systolic blood pressure
diastolicbloodpressure= 80 # Diastolic blood pressure
cholesterol= int(input('Enter the Cholesterol (1:Normal 2:Above Normal 3:Well Above Normal)')) # 1: normal, 2: above normal, 3: well above normal
gluc= int(input('Enter Glucose (1:Normal 2:Above Normal 3:Well Above Normal)')) # 1: normal, 2: above normal, 3: well above normal
smoke= int(input('Enter the Smoking Habit (0: Smoke 1:Doesnt Smoke)')) # 1 if you smoke, 0 if not
alco= int(input('Enter the Alcoholic Habit (0:Alcholic 1:Not Alcholoic)')) # 1 if you drink alcohol, 0 if not
active= int(input('Enter the Physical Acitivity (0:Physically Active 1:Physically Not Active)')) # 1 if you do physical activity, 0 if not



agedayscale=(age-df["age"].min())/(df["age"].max()-df["age"].min())
heightscale=(height-df["height"].min())/(df["height"].max()-df["height"].min())
weightscale=(weight-df["weight"].min())/(df["weight"].max()-df["weight"].min())
sbpscale=(systolicbloodpressure-df["ap_hi"].min())/(df["ap_hi"].max()-df["ap_hi"].min())
dbpscale=(diastolicbloodpressure-df["ap_lo"].min())/(df["ap_lo"].max()-df["ap_lo"].min())
cholesterolscale=(cholesterol-df["cholesterol"].min())/(df["cholesterol"].max()-df["cholesterol"].min())
glucscale=(gluc-df["gluc"].min())/(df["gluc"].max()-df["gluc"].min())

single=np.array([agedayscale, gender, heightscale, weightscale, sbpscale, dbpscale, cholesterolscale, glucscale, smoke, alco, active ])
singledf=pd.DataFrame(single)
final=singledf.transpose()



##Predictiom
weights_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/Weights/weights-improvement-03-1.00.hdf5'
model.load_weights(weights_path)
result = model.predict(final)

print('The Probablity is :{}'.format(result))
if(result[0][0] > 0.7):
    print('Has a cardiovascular Disease')
else:
    print('Doesnt Have a Cardiovascular Disease')
