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
import time
start = time.time()

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

for i in file_list_py01:
    data = pd.read_csv(path + i)
    df01 = pd.concat([df01, data])

Label_enc = sklearn.preprocessing.LabelEncoder()
df['시간'] = Label_enc.fit_transform(df['시간'])
df01['시간'] = Label_enc.fit_transform(df['시간'])

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
data_height = 150
data_width = 150
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
        image = cv2.resize(image, (data_height, data_width))/255
        images[n, :, :, :] =image
    
    label = np.array(label)
        
    return (label, images)

(label, images) = make_file(data_height, data_width, channel_n, batch_size)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        rotation_range = 180,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image 
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

images01 = images.reshape(-1, 150 * 150 * 3)

from sklearn.preprocessing import LabelEncoder
items = label
encoder = LabelEncoder()
encoder.fit(items)
label = encoder.transform(items)

scaler = StandardScaler()
from sklearn.decomposition import PCA
result02 = pd.DataFrame()
for i in range(0, 1592):
    file_name[i] = file_name[i].fillna(0)
    result01 = scaler.fit_transform(file_name[i])
    data_scaled = pd.DataFrame(result01, columns=result01[0])
    data_scaled.describe()
    pca = PCA(n_components=1)
    result01 = pca.fit_transform(data_scaled)
    result01 = pd.DataFrame(result01, columns=["시간"])
    result01 = result01.T
    result02 = pd.concat([result02, result01])

test02 = pd.DataFrame()
for i in range(0, 460):
    file_name01[i] = file_name01[i].fillna(0)
    test01 = scaler.fit_transform(file_name01[i])
    data_scaled = pd.DataFrame(test01, columns=test01[0])
    data_scaled.describe()
    pca = PCA(n_components=1)
    test01 = pca.fit_transform(data_scaled)
    test01 = pd.DataFrame(test01, columns=["시간"])
    test01 = test01.T
    test02 = pd.concat([test02, test01])

images01 = pd.DataFrame(images01)
result03 = result02.reset_index(drop = True)
result03 = pd.concat([result03, images01], axis = 1)

data_dir = os.chdir("E:/dacon_data/kist01/kist/flip_01")
data_list = glob('*.*')
data_height = 150
data_width = 150
channel_n = 3
batch_size = len(data_list)
print(data_list)
(label02, flip01) = make_file(data_height, data_width, channel_n, batch_size)
flip01_01 = flip01.reshape(-1, 150 * 150 * 3)
flip01_01 = pd.DataFrame(flip01_01)
aug01 = result02.reset_index(drop = True)
flip_result01 = pd.concat([aug01, flip01_01], axis = 1)
flip02_01 = np.empty(shape = [1, 150 * 150 * 3])
flip03_01 = np.empty(shape = [1, 150 * 150 * 3])
flip04_01 = np.empty(shape = [1, 150 * 150 * 3])
flip05_01 = np.empty(shape = [1, 150 * 150 * 3])

for i in range(0, 1592):
    flip180 = cv2.rotate(images[i, :, :, :], cv2.ROTATE_180)
    flip02 = flip180.reshape(-1, 150 * 150 * 3)
    flip02_01 = np.concatenate((flip02_01, flip02), axis = 0)
flip02_01 = np.delete(flip02_01, 0, axis =0)
flip02_01 = pd.DataFrame(flip02_01)
aug02 = result02.reset_index(drop = True)
flip_result02 = pd.concat([aug02, flip02_01], axis = 1)

for i in range(0, 1592):
    flip270 = cv2.rotate(images[i, :, :, :], cv2.ROTATE_90_COUNTERCLOCKWISE)
    flip03 = flip270.reshape(-1, 150 * 150 * 3)
    flip03_01 = np.concatenate((flip03_01, flip03), axis = 0)
flip03_01 = np.delete(flip03_01, 0, axis =0)
flip03_01 = pd.DataFrame(flip03_01)
aug03 = result02.reset_index(drop = True)
flip_result03 = pd.concat([aug03, flip03_01], axis = 1)

for i in range(0, 1592):
    flip_flip = cv2.flip(images[i, :, :, :], 1)
    flip04 = flip_flip.reshape(-1, 150 * 150 * 3)
    flip04_01 = np.concatenate((flip04_01, flip04), axis = 0)
flip04_01 = np.delete(flip04_01, 0, axis =0)
flip04_01 = pd.DataFrame(flip04_01)
aug04 = result02.reset_index(drop = True)
flip_result04 = pd.concat([aug04, flip04_01], axis = 1)

for i in range(0, 1592):
    flip_flip_01 = cv2.flip(images[i, :, :, :], 0)
    flip05 = flip_flip_01.reshape(-1, 150 * 150 * 3)
    flip05_01 = np.concatenate((flip05_01, flip05), axis = 0)
flip05_01 = np.delete(flip05_01, 0, axis =0)
flip05_01 = flip05.reshape(-1, 150 * 150 * 3)
flip05_01 = pd.DataFrame(flip05_01)
aug05 = result02.reset_index(drop = True)
flip_result05 = pd.concat([aug05, flip05_01], axis = 1)

final_result05 = pd.concat([result03, flip_result01, flip_result02, flip_result03, flip_result04, flip_result05])

weight01 = np.concatenate(weight).tolist()
weight_01 = weight01.copy()
weight_02 = weight01.copy()
weight_03 = weight01.copy()
weight_04 = weight01.copy()
weight_05 = weight01.copy()
weight02 = weight01 + weight_01 + weight_02 + weight_03 + weight_04 + weight_05
final_result05 = final_result05.fillna(0)

x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(final_result05,weight02,test_size=0.3, shuffle=True, random_state=0)

model = sklearn.neural_network.MLPRegressor(
hidden_layer_sizes = (128, 16),
activation = 'relu',
solver = 'adam',
learning_rate_init = 0.001,
max_iter = 500,
batch_size = 16,
alpha = 0.0001,
warm_start = False,
random_state = 0 )

model.fit(x_train, y_train)
print(model.score(x_test, y_test))

data_dir = os.chdir("E:/dacon_data/kist/test/image")
data_list = glob('*.*')
data_height = 150
data_width = 150
channel_n = 3
batch_size = len(data_list)
(label01, images_test) = make_file(data_height, data_width, channel_n, batch_size)
images_test01 = images_test.reshape(-1, 150 * 150 * 3)
images_test01 = pd.DataFrame(images_test01)
test03 = test02.reset_index(drop = True)
test03 = pd.concat([test03, images_test01], axis = 1)

print("time :", time.time() - start)

a = model.predict(test03)
submission.iloc[:, 1] = a
submission.to_csv('E:/dacon_data/kist/sample_submission_04.csv',index=False)
