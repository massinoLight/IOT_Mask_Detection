from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
 
import os # operating system
 
import cv2 # open cv
import matplotlib.pyplot as plt
 
batch_size = 64
img_height = 200
img_width = 200
data_dir = "mini_dataset"
# Dataset d'entraînement
train_data = tf.keras.preprocessing.image_dataset_from_directory(
 data_dir,
 class_names=['mini_face_dataset', 'mini_masked_face_dataset'],
 validation_split=0.2,
 subset="training",
 seed=42,
 image_size=(img_height, img_width),
 batch_size=batch_size,
 )
 
# Dataset de validation
val_data = tf.keras.preprocessing.image_dataset_from_directory(
 data_dir,
 class_names=['mini_face_dataset', 'mini_masked_face_dataset'],
 validation_split=0.2,
 subset="validation",
 seed=42,
 image_size=(img_height, img_width),
 batch_size=batch_size,
 )
 
num_classes = 2
 
# CNN Maison de classification binaire
model = tf.keras.Sequential([
   layers.experimental.preprocessing.Rescaling(1./255),
   layers.Conv2D(64,4, activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(32,4, activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(16,4, activation='relu'),
   layers.MaxPooling2D(),
   layers.Flatten(),
   layers.Dense(64,activation='relu'),
   layers.Dense(num_classes, activation='softmax')
])
 
model.compile(optimizer='adam',
             loss=tf.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'],)
 
logdir="logs"
 
# Visualisation des métriques de validation
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir,
                                                  embeddings_data=train_data)
# Entraînement du modèle
model.fit(
   train_data,
 validation_data=val_data,
 epochs=12,
 callbacks=[tensorboard_callback]
)
 
from keras.models import model_from_json
 
# Sauvegarde du modèle en json
model_json = model.to_json()
with open("mask_detector.json", "w") as json_file:
   json_file.write(model_json)
# Sauvegarde des paramètres entraînable du modèles
model.save_weights("mask_detector.h5")
print("Saved model to disk")


