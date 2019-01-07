import keras as keras
from keras.models import Model
import keras.backend as K
from keras import callbacks
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.utils import to_categorical
import h5py
from sklearn.preprocessing import MultiLabelBinarizer

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
from urllib.request import urlretrieve

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
pre_model = VGG16()

def create_model(pre_model):
    x = pre_model.layers[-2].output
  
    # add output layer
    predictions = Dense(2, activation='softmax')(x)
    
    # initialize Model (functional API)
    model = Model(inputs=pre_model.input, outputs=predictions)
        
    # freezes hidden layers
    for layer in model.layers[:-5]: 
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    
    return model


def preprocess(url):
    img_path = '/tmp/image.jpg'
    urlretrieve(url, img_path)

    img = image.load_img(img_path, target_size=(224, 224)) # PIL img
    x = image.img_to_array(img) # (224, 224, 3)
    x = np.expand_dims(x, axis=0) # (1, 224, 224, 3)
    x = preprocess_input(x)

    return x

def main(job_dir,**args):

    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'

    ##Loading the data
    df = pd.read_csv('https://www.dropbox.com/s/iw2zspr6mc3c9rl/dogcat.csv?dl=1')

    X_array_processed = []
    y_array = []

    for index, row in df.iterrows():
        try:
            vectorized_img = preprocess(row[0])
            label = row[1] # label(s)

            X_array_processed.append(vectorized_img)
            y_array.append(label)
            
        except Exception as e: # img not available (e.g. 404)
            #print(e)
            continue

    y_array_2 = []

    for elem in y_array:
        y_array_2.append(elem[0])
    
    int_array = pd.factorize(y_array_2)[0]

    one_hot_array = to_categorical(int_array, num_classes=14)
    y_array_processed = one_hot_array

    new = []
    for i in y_array_processed:
        new.append(np.asarray(i))

    y_array_processed_np = new

    X_train, X_val, y_train, y_val = train_test_split(X_array_processed, 
                                                      y_array_processed_np, 
                                                      test_size=0.3,
                                                      random_state=42)

    ##Using the GPU
    with tf.device('/device:GPU:0'):

        ## Initializing the model
        model = create_model(pre_model)

        # callbacks
        # callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2)]
        #TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        ##fitting the model
        model.fit(x=np.array(X_train),
                y=np.array(y_train),
                batch_size=32,
                epochs=10,
                verbose=1,
                #callbacks=callbacks,
                validation_data=(np.array(X_val), np.array(y_val)))

        # Save model.h5 on to google storage
        model.save('model.h5')

        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
