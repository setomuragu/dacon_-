import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import sklearn
from sklearn import *
from glob import glob
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
from scipy import stats
import time
start = time.time()
print(start)

submission = pd.read_csv('E:/dacon_data/kist/sample_submission.csv')
path = 'E:/dacon_data/kist01/kist/train/meta/'
path01 = 'E:/dacon_data/kist/test/meta/'
file_list = os.listdir(path)
file_list01 = os.listdir(path01)
file_list_py = [file for file in file_list if file.endswith('.csv')]
file_list_py01 = [file for file in file_list if file.endswith('.csv')]

file_name = []
file_name01 = []
for file in file_list:
    if file.count(".") == 1:
        name = file.split('.')[0]
        file_name.append(name)
    else:
        for k in range(len(file) -1, 0, -1):
            if file[k] == '.':
                file_name.append(file[:k])
                break
            
for file in file_list01:
    if file.count(".") == 1:
        name = file.split('.')[0]
        file_name01.append(name)
    else:
        for k in range(len(file) -1, 0, -1):
            if file[k] == '.':
                file_name01.append(file[:k])
                break

df = pd.DataFrame()
df01 = pd.DataFrame()
for i in file_list_py:
    data = pd.read_csv(path + i)
    df = pd.concat([df, data])

for i in file_list01:
    data = pd.read_csv(path01 + i)
    df01 = pd.concat([df01, data])

df = df.drop(columns = ['CO2관측치', 'EC관측치', '최근분무량', '블루 LED동작강도', '냉방부하', '난방온도', '청색광추정광량', '외부온도관측치'])
df01 = df01.drop(columns = ['CO2관측치', 'EC관측치', '최근분무량', '블루 LED동작강도', '냉방부하', '난방온도', '청색광추정광량', '외부온도관측치'])

Label_enc = sklearn.preprocessing.LabelEncoder()
df['시간'] = Label_enc.fit_transform(df['시간'])
df01['시간'] = Label_enc.fit_transform(df01['시간'])

for i in range(0, 1592):
    file_name[i] = df.iloc[0 + i * 1440 : 1440 + i * 1440, :]
for i in range(0, 460):
    file_name01[i] = df01.iloc[0 + i * 1440 : 1440 + i * 1440, :]

list01 = []
leaf = pd.DataFrame()
main_path = 'E:/dacon_data/kist/train/*'
main_folder = r'E:/dacon_data/kist/train/'
for item in os.listdir(main_folder):
    sub_folder = os.path.join(main_folder, item)
    if os.path.isdir(sub_folder):
        list01.append(sub_folder)

data_list = []
data_list01 = []
data_list02 = []
label_list = []
for i in list01:
    label_leaf = pd.read_csv(i + '/label.csv')
    leaf = pd.concat([leaf, label_leaf])

for i in range(0, 1592):
    data01 = file_name[i].to_numpy().reshape(1, 15840)
    data_list01.append(data01)

for i in range(0, 460):
    data01 = file_name01[i].to_numpy().reshape(1, 15840)
    data_list02.append(data01)

list01 = []
leaf = pd.DataFrame()
main_path = 'E:/dacon_data/kist/train/*'
main_folder = r'E:/dacon_data/kist/train/'
for item in os.listdir(main_folder):
    sub_folder = os.path.join(main_folder, item)
    if os.path.isdir(sub_folder):
        list01.append(sub_folder)

label_list = []
for i in list01:
    label_leaf = pd.read_csv(i + '/label.csv')
    leaf = pd.concat([leaf, label_leaf])

leaf_weight = leaf.iloc[:, 1]
weight = leaf_weight.to_numpy()
weight = weight.reshape(-1, 1)

data_dir = os.chdir("E:/dacon_data/kist01/kist/train")
import glob
from glob import glob
data_list = glob('*.*')
data_height = 154
data_width = 205
channel_n = 3
batch_size = len(data_list)

# data processing 함수 
from tensorflow.keras.preprocessing.text import text_to_word_sequence

def make_file(data_height, data_width, channel_n, batch_size):
    label = []
    images = np.zeros((batch_size, data_height, data_width, channel_n))
    for n, path in enumerate(data_list[:batch_size]):

    # lable 
        token = text_to_word_sequence(data_list[n])
        label.append(token[0])
        
    # image transform
        image = cv2.imread(data_list[n])
        image = cv2.resize(image, (data_width, data_height))/255
        images[n, :, :, :] =image
    
    label = np.array(label)
        
    return (label, images)

(label, images) = make_file(data_height, data_width, channel_n, batch_size)

images01 = images.reshape(-1, 154 * 205 * 3)

from sklearn.preprocessing import LabelEncoder
items = label
encoder = LabelEncoder()
encoder.fit(items)
label = encoder.transform(items)

scaler = StandardScaler()

data_list01 = np.array(data_list01).reshape(1592, 15840)
data_list02 = np.array(data_list02).reshape(460, 15840)

images01 = pd.DataFrame(images01)
train02 = pd.DataFrame(data_list01)
test02 = pd.DataFrame(data_list02)
train02 = train02.fillna(0)
test02 = test02.fillna(0)


result03 = pd.concat([train02, images01], axis = 1)

flip04_01 = np.empty(shape = [1, 154 * 205 * 3])
flip05_01 = np.empty(shape = [1, 154 * 205 * 3])

for i in range(0, 1592):
    flip_flip = cv2.flip(images[i, :, :, :], 1)
    flip04 = flip_flip.reshape(-1, 154 * 205 * 3)
    flip04_01 = np.concatenate((flip04_01, flip04), axis = 0)
flip04_01 = np.delete(flip04_01, 0, axis =0)
flip04_01 = pd.DataFrame(flip04_01)

flip_result04 = pd.concat([train02, flip04_01], axis = 1)

for i in range(0, 1592):
    flip_flip = cv2.flip(images[i, :, :, :], 0)
    flip05 = flip_flip.reshape(-1, 154 * 205 * 3)
    flip05_01 = np.concatenate((flip05_01, flip05), axis = 0)
flip05_01 = np.delete(flip05_01, 0, axis =0)
flip05_01 = pd.DataFrame(flip05_01)

flip_result05 = pd.concat([train02, flip05_01], axis = 1)


final_result05 = pd.concat([result03, flip_result04, flip_result05])

weight01 = np.concatenate(weight).tolist()
weight_01 = weight01.copy()
weight_02 = weight01.copy()
weight02 = weight01 + weight_01 + weight_02
final_result05 = final_result05.fillna(0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
weight02[:] = scaler.fit_transform(weight02[:])
final_result05[:] = scaler.fit_transform(final_result05[:])

x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(final_result05,weight02,test_size=0.3, shuffle=True, random_state=0)

from sklearn import svm
model = svm.SVR()

model.fit(x_train, y_train)
print(model.score(x_test, y_test))


data_dir = os.chdir("E:/dacon_data/kist/test/image")
data_list = glob('*.*')
data_height = 154
data_width = 205
channel_n = 3
batch_size = len(data_list)
(label01, images_test) = make_file(data_height, data_width, channel_n, batch_size)
images_test01 = images_test.reshape(-1, 154 * 205 * 3)
images_test01 = pd.DataFrame(images_test01)
test_data = pd.concat([test02, images_test01], axis = 1)

print("time :", time.time() - start)

test_data[:] = scaler.fit_transform(test_data[:])
a = model.predict(test_data)
submission.iloc[:, 1] = a
submission.to_csv('E:/dacon_data/kist/sample_submission_06.csv',index=False)
