#!/usr/bin/env python
# coding: utf-8

# In[11]:


import librosa
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, UpSampling2D, MaxPooling2D
from keras.models import Model
import librosa
import librosa.display
import matplotlib
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint
import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans


# In[139]:



def create_mel_spec_imagedata(song_path, save_image_path):
    for audioname in os.listdir(song_path):

        audio_path= "/home/arpita/Hw4/dataset/" + audioname #+ '.wav' #location
        y, sr= librosa.load(audio_path)   
        plt.axis('off') # no axis
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        S = librosa.feature.melspectrogram(y= y , sr= sr)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.axis('off')
        plt.margins(0,0)
        name = audioname.split(".")[0]
        plt.savefig(save_image_path + name + ".png", bbox_inches=None, pad_inches=0, aspect = 'auto')
        plt.close()
    print("Saved all images at ", save_image_path)


# In[140]:


create_mel_spec_imagedata("/home/arpita/Hw4/dataset/", "/home/arpita/Hw4/audioimages/melspec/")


# In[141]:


def load_imgs_dataset(path):
    img_dataset = []
    for i in range(1, 672):
        imgname = str(i) + ".png"
        img = cv2.imread(path + imgname, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_dataset.append(img)
    img_dataset = np.array(img_dataset)
    print(img_dataset.shape)
    return img_dataset


# In[144]:


data_set_load = load_imgs_dataset("/home/arpita/Hw4/audioimages/melspec/")
print(data_set_load.shape)

#block size of 72

w=data_set_load[0].shape[0]
h=data_set_load[0].shape[1]
block_data_set = []

for img in data_set_load:
    for r in range(0, w, 72):
        for c in range(0, h, 72):
            block_data_set.append(img[r:r+72,c:c+72]) 

block_data_set = np.array(block_data_set)
print(block_data_set.shape)


# In[135]:


input_img = Input(shape=(72, 72, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

extraction_model = Model(input_img, decoded)
extraction_model.compile(optimizer='adam', loss='MSE')

extraction_model.summary()


# In[ ]:



checkpointer = ModelCheckpoint(filepath = "/home/arpita/Hw4/Models/PR4.ckpt", 
                               monitor='loss',
                               save_best_only=True)

extraction_model.fit(block_data_set, block_data_set,
                     batch_size=64, 
                     epochs=200, 
                     callbacks=[checkpointer])


# In[146]:


encoder = Model(input_img, encoded)
extraction_model.load_weights(filepath ="/home/arpita/Hw4/Models/PR4.ckpt")

for i in range(0, len(encoder.layers)):

    extracted_weights = extraction_model.layers[i].get_weights()
    encoder.layers[i].set_weights(extracted_weights)
    print(extracted_weights)


# In[151]:


encoder_full_pred = encoder.predict(block_data_set)
print(encoder_full_pred.shape)
reduced_feature_vector = encoder_full_pred.reshape(671, 1944)
print(reduced_feature_vector.shape)
kmeans = KMeans(n_clusters=20)
kmeans.fit(reduced_feature_vector)
kmeans.cluster_centers_
clusters = kmeans.predict(reduced_feature_vector)
prediction = kmeans.labels_ + 1
print(prediction)


# In[152]:


filename = "/home/arpita/Hw4/Results/Prediction_file.txt"

with open(filename, 'w', newline='') as w1:
    writer = csv.writer(w1, delimiter=' ')
    for p in prediction:
        writer.writerow([p])


# In[ ]:


x_test_encoded = encoder.predict(block_data_set, batch_size=64)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

