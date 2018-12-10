
# coding: utf-8

# In[1]:


# For mnist data
# from keras.datasets import mnist
# For one-hot shape
from keras.utils import np_utils
# For Sequential model
from keras.models import Sequential
# For defining optimizers for each models
from keras import optimizers
# For model implemetations
from keras.layers import Dense, Activation, LSTM
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
# To check channels_last vs channels_first
from keras import backend as K
# For TensorBoard
from keras.callbacks import TensorBoard, ModelCheckpoint


# In[2]:


import json
from pprint import pprint
import numpy as np

# Changmin Seo, Ke Yin

def read_json_data(filename, shuffle, use_angle):
    """This function is for reading json data file which contains pose estimation output"""
    data = json.load(open(filename))

    xy_idx = np.ones(54, dtype=bool);
    for i in range(54):
        if i%3 == 2:
            xy_idx[i] = 0
    x_idx = np.zeros(54, dtype=bool);
    for i in range(54):
        if i%3 == 0:
            x_idx[i] = 1
    y_idx = np.zeros(54, dtype=bool);
    for i in range(54):
        if i%3 == 1:
            y_idx[i] = 1

    frame_per_step = 90

    count = 0

    X_list = []
    y_list = []
    X_val_list = []
    y_val_list = []
    X_test_list = []
    y_test_list = []
    for key in data.keys():
        for key2 in data[key].keys():
            count = count + 1
            # pose estimation data here are not stable, so it should be excluded
            if key == 'W3' and key2 == '8L':
                continue
            if key == 'W3' and key2 == '9L':
                continue


            if key2 == '1R' or key2 == '2R' or key2 == '3R' or key2 == '4R':
                right_flag = True
            else:
                right_flag = False

            if key2 == '13' or key2 == '9L':
                val_data_flag = True
                print(key, key2, 'count= ', count, ' validation')
            else:
                val_data_flag = False
            if  key2 == '7L' or key2 == '15':
                test_data_flag = True
                print(key, key2, 'count= ', count, ' test')
            else:
                test_data_flag = False
            if (val_data_flag or test_data_flag) == False:
                print(key, key2, 'count= ', count, ' train')

            for key3 in data[key][key2].keys():
                if key3 == 'poses':
                    arr = np.array(data[key][key2][key3])
                    if arr.shape[0] != 0:
                        if right_flag:
                            arr[:,x_idx] *= -1
                            arr[:,x_idx] +=270
                        # normalize
                        x_means = np.mean(arr[:,x_idx], axis = 0)
                        y_means = np.mean(arr[:,y_idx], axis = 0)
                        x_mean_video = np.mean(x_means)
                        y_mean_video = np.mean(y_means)
                        x_std_video = np.std(x_means)
                        y_std_video = np.std(y_means)
                        arr[:,x_idx] -= x_mean_video
                        arr[:,y_idx] -= y_mean_video
                        arr[:,x_idx] /= x_std_video
                        arr[:,y_idx] /= y_std_video

                        arr_ori = np.array(data[key][key2][key3])

                        if use_angle:
                            for i in range(arr.shape[0] - frame_per_step + 1):
                                if val_data_flag:
                                    X_val_list.append(get_angle(arr_ori[i:i+frame_per_step,x_idx], arr_ori[i:i+frame_per_step,y_idx]))
                                    y_val_list.append(data[key][key2]['label'])
                                elif test_data_flag:
                                    X_test_list.append(get_angle(arr_ori[i:i+frame_per_step,x_idx], arr_ori[i:i+frame_per_step,y_idx]))
                                    y_test_list.append(data[key][key2]['label'])
                                else:
                                    X_list.append(get_angle(arr_ori[i:i+frame_per_step,x_idx], arr_ori[i:i+frame_per_step,y_idx]))
                                    y_list.append(data[key][key2]['label'])
                        else:
                            for i in range(arr.shape[0] - frame_per_step + 1):
                                if val_data_flag:
                                    X_val_list.append(arr[i:i+frame_per_step,xy_idx])
                                    y_val_list.append(data[key][key2]['label'])
                                elif test_data_flag:
                                    X_test_list.append(arr[i:i+frame_per_step,xy_idx])
                                    y_test_list.append(data[key][key2]['label'])
                                else:
                                    X_list.append(arr[i:i+frame_per_step,xy_idx])
                                    y_list.append(data[key][key2]['label'])

    # list to np array
    X_data = np.array(X_list)
    y_data = np.array(y_list)
    X_val_data = np.array(X_val_list)
    y_val_data = np.array(y_val_list)
    X_test_data = np.array(X_test_list)
    y_test_data = np.array(y_test_list)
    print(X_data.shape)
    print(y_data.shape)
    print(X_val_data.shape)
    print(y_val_data.shape)
    print(X_test_data.shape)
    print(y_test_data.shape)
    print('Numpy array data generated.')

    # Shuffle train and validation data
    if (shuffle == True):
        print('Now shuffle the data.')
        p = np.random.permutation(len(X_data))
        X_data = X_data[p]
        y_data = y_data[p]
        p = np.random.permutation(len(X_val_data))
        X_val_data = X_val_data[p]
        y_val_data = y_val_data[p]
        print('Shuffle finished.')

    print(X_data.shape)
    print(y_data.shape)
    print(X_val_data.shape)
    print(y_val_data.shape)
    print(X_test_data.shape)
    print(y_test_data.shape)

    return X_data, y_data, X_val_data, y_val_data, X_test_data, y_test_data


# In[3]:


# Changmin Seo

def get_cos_angle(x_array, y_array):
    """This function is for calculating angles between each bone from the openpose skelleton"""
    vectors = get_vectors(x_array, y_array)
#     angles_vector_idx = [[0,1],[1,7],[1,2],[2,3],[0,4],[4,10],[4,5],[5,6],[7,8],[8,9],[10,11],[11,12]]
    angles_vector_idx = [[1,7],[4,10],[7,8],[8,9],[10,11],[11,12]]
    for i in range(len(angles_vector_idx)):
        curr = get_joint_cos_angle(vectors, angles_vector_idx[i])

        if i == 0:
            prev = curr
        else:
            prev = np.column_stack((prev, curr))
    return np.arccos(prev)

# Changmin Seo

def get_joint_cos_angle(vectors, pair):
    """This function is for calculating an angle betwwen two bones from the openpose skelleton"""
    u = vectors[pair[0]]
    v = vectors[pair[1]]
    dot = np.add(np.multiply(u[0],v[0]), np.multiply(u[1],v[1]))
    u_norm = np.sqrt(np.multiply(u[0],u[0]), np.multiply(u[1],u[1]))+1e-5
    v_norm = np.sqrt(np.multiply(v[0],v[0]), np.multiply(v[1],v[1]))+1e-5
    result = np.divide(np.divide(dot, u_norm), v_norm)
    result[result > 1] = 1
    result[result < -1] = -1
    return result

# Changmin Seo

def get_vectors(x_array, y_array):
    """This function is for calculating vectors of bones from the openpose skelleton"""
    vector_joint_idx = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]
    vectors = []
    for i in range(len(vector_joint_idx)):
        vectors.append(get_vector(x_array, y_array, vector_joint_idx[i]))
    return vectors

# Changmin Seo

def get_vector(x_array, y_array, pair):
    """This function is for calculating a vector of a bone from the openpose skelleton"""
    x = x_array[:,pair[0]]-x_array[:,pair[1]]
    y = y_array[:,pair[0]]-y_array[:,pair[1]]
    return [x, y]


# In[4]:


# Changmin Seo

def ang_vec(v1_arr, v2_arr):
    """This function is for calculating vectors of bones from the openpose skelleton, a re-implementation"""
    v1_norm = np.linalg.norm(v1_arr, axis=1)+1e-5
    v2_norm = np.linalg.norm(v2_arr, axis=1)+1e-5
    ang_arr = np.arccos( np.sum(v1_arr*v2_arr, axis=1)/v1_norm/v2_norm )
    result = ang_arr.reshape(-1,1)

    return result

# Changmin Seo

def get_angle(x_array, y_array):
    """This function is for calculating angles between each bone from the openpose skelleton, a re-implementation"""
    x_arr = x_array.reshape(-1, 1, 18)
    y_arr = y_array.reshape(-1, 1, 18)
    xy_arr = np.concatenate([x_arr, y_arr], axis=1)

    v_arr_list = []
    v_mag_list = []
    ang_arr_list = []

    v_arr_list.append(xy_arr[:,:,1]-xy_arr[:,:,0]) # v0
    v_arr_list.append(xy_arr[:,:,2]-xy_arr[:,:,1]) # v1
    v_arr_list.append(xy_arr[:,:,3]-xy_arr[:,:,2]) # v2
    v_arr_list.append(xy_arr[:,:,4]-xy_arr[:,:,3]) # v3
    v_arr_list.append(xy_arr[:,:,5]-xy_arr[:,:,1]) # v4
    v_arr_list.append(xy_arr[:,:,6]-xy_arr[:,:,5]) # v5
    v_arr_list.append(xy_arr[:,:,7]-xy_arr[:,:,6]) # v6
    v_arr_list.append(xy_arr[:,:,8]-xy_arr[:,:,1]) # v7
    v_arr_list.append(xy_arr[:,:,9]-xy_arr[:,:,8]) # v8
    v_arr_list.append(xy_arr[:,:,10]-xy_arr[:,:,9]) # v9
    v_arr_list.append(xy_arr[:,:,11]-xy_arr[:,:,1]) # v10
    v_arr_list.append(xy_arr[:,:,12]-xy_arr[:,:,11]) # v11
    v_arr_list.append(xy_arr[:,:,13]-xy_arr[:,:,12]) # v12

    ang_arr_list.append( ang_vec(v_arr_list[0], v_arr_list[1]) ) # ang0
    ang_arr_list.append( ang_vec(v_arr_list[1], v_arr_list[7]) ) # ang1
    ang_arr_list.append( ang_vec(v_arr_list[1], v_arr_list[2]) ) # ang2
    ang_arr_list.append( ang_vec(v_arr_list[2], v_arr_list[3]) ) # ang3
    ang_arr_list.append( ang_vec(v_arr_list[0], v_arr_list[4]) ) # ang4
    ang_arr_list.append( ang_vec(v_arr_list[4], v_arr_list[10]) ) # ang5
    ang_arr_list.append( ang_vec(v_arr_list[4], v_arr_list[5]) ) # ang6
    ang_arr_list.append( ang_vec(v_arr_list[5], v_arr_list[6]) ) # ang7
    ang_arr_list.append( ang_vec(v_arr_list[7], v_arr_list[10]) ) # ang8
    ang_arr_list.append( ang_vec(v_arr_list[7], v_arr_list[8]) ) # ang9
    ang_arr_list.append( ang_vec(v_arr_list[8], v_arr_list[9]) ) # ang10
    ang_arr_list.append( ang_vec(v_arr_list[10], v_arr_list[11]) ) # ang11
    ang_arr_list.append( ang_vec(v_arr_list[11], v_arr_list[12]) ) # ang12

    result = np.concatenate(ang_arr_list, axis=1)
    return result


# In[5]:


# Changmin Seo, Ke Yin

def data_preprocess(X_train, y_train, X_validation, y_validation, X_test, y_test):
    """This function is for preprocessing data. It returns processed data and data related parameters"""

    # Define input and output dimension for mlp
    input_dim = X_train.shape[1] * X_train.shape[2]
    output_dim = 4

    # Define input shpae for lstm
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Reshape X data to fit as the input_dim for mlp model
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_validation = X_validation.astype('float32')
    X_train_mlp = X_train.reshape(X_train.shape[0], input_dim)
    X_validation_mlp = X_validation.reshape(X_validation.shape[0], input_dim)
    X_test_mlp = X_test.reshape(X_test.shape[0], input_dim)

    # Change the label to one-hot shape
    y_train = np_utils.to_categorical(y_train, output_dim)
    y_test = np_utils.to_categorical(y_test, output_dim)
    y_validation = np_utils.to_categorical(y_validation, output_dim)

    return X_train, X_train_mlp, y_train, X_validation, X_validation_mlp, y_validation, X_test, X_test_mlp, y_test, input_dim, output_dim, input_shape


# In[6]:


# Changmin Seo

def get_model(model_class):
    """This functio nis for building the model for mlp and lstm"""
    model = Sequential()
    if model_class == 'mlp' or model_class == 'mlp_angle' :
        # Add a hidden layer with ReLU activation
        model.add(Dense(1024, input_dim=input_dim,                         activation='relu'))

        # Add a output layer with softmax activation
        model.add(Dense(output_dim, activation='softmax'))

        # Define adam optimizer as example
        optimizer = optimizers.Adam(lr=1e-4)

# Ke Yin

    elif model_class == 'lstm' or model_class == 'lstm_angle':
        # add LSTM layer with input shape
        model.add(LSTM(256,return_sequences=True,input_shape=input_shape))
        # add dropout
        model.add(Dropout(0.5))
        #add another LSTM layer
        model.add(LSTM(256))
        # add dropout
        model.add(Dropout(0.5))
        # add output layer with softmax activation
        model.add(Dense(output_dim, activation='softmax'))

        # Define adam optimizer as example
        optimizer = optimizers.Adam(lr=1e-4)

# Changmin Seo, Ke Yin

    # Complie the model
    model.compile(optimizer=optimizer,                   loss='categorical_crossentropy',                   metrics=['accuracy'])
    return model

# Changmin Seo

def train_model(model_class, model, batch_size, epochs):
    """This function is for training each models"""
    # Assign X data according to model class
    # Define the file directory for each model class
    if model_class == 'mlp':
        X_train_ = X_train_mlp
        X_validation_ = X_validation_mlp
        filedir = './mtn/mlp'
    elif model_class == 'mlp_angle':
        X_train_ = X_train_mlp
        X_validation_ = X_validation_mlp
        filedir = './mtn/mlp_angle'
# Ke Yin

    elif model_class == 'lstm':
        X_train_ = X_train
        X_validation_ = X_validation
        filedir = './mtn/lstm'
    elif model_class == 'lstm_angle':
        X_train_ = X_train
        X_validation_ = X_validation
        filedir = './mtn/lstm_angle'

# Changmin Seo

    # Define the callbacks
    # TensorBoard allows us to record and plot log
    # ModelCheckpoint can save model after each epoch
    callbacks = [
        TensorBoard(log_dir=filedir, \
            histogram_freq=0, batch_size=batch_size, \
            write_graph=True, write_grads=False, \
            write_images=True, embeddings_freq=0, \
            embeddings_layer_names=None, \
            embeddings_metadata=None),
        ModelCheckpoint(filedir + '/weights.best.hdf5', \
            monitor='val_loss', verbose=0, \
            save_best_only=True, save_weights_only=False, \
            mode='auto', period=1)
    ]

    # Train the model
    history = model.fit(X_train_, y_train,                         batch_size=batch_size,                         epochs=epochs, verbose=1,                         validation_data=(X_validation_,                                          y_validation),                         callbacks=callbacks)
# Changmin Seo, Ke Yin

def evaluate_model(model_class, model):
    """This function is for evaluating the model"""
    # Assign X data according to model class
    if model_class == 'mlp' or model_class == 'mlp_angle':
        X_test_ = X_test_mlp
    elif model_class == 'lstm' or model_class == 'lstm_angle':
        X_test_ = X_test

    # evaluate the model
    score = model.evaluate(X_test_, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# Changmin Seo

def load_model(model_class):
    """This function is for loading the saved model"""
    # Define the file directory for each model
    if model_class == 'mlp':
        filedir = './mtn/mlp'
    elif model_class == 'mlp_angle':
        filedir = './mtn/mlp_angle'
    elif model_class == 'lstm':
        filedir = './mtn/lstm'
    elif model_class == 'lstm_angle':
        filedir = './mtn/lstm_angle'

    # Define the filename
    filename = 'weights.best.hdf5'

    # Build the model
    model = Sequential()
    if model_class == 'mlp' or model_class == 'mlp_angle':
        # Add a hidden layer with ReLU activation
        model.add(Dense(1024, input_dim=input_dim,                         activation='relu'))

        # Add a output layer with softmax activation
        model.add(Dense(output_dim, activation='softmax'))

        # Define adam optimizer as example
        optimizer = optimizers.Adam(lr=1e-4)

# Ke Yin

    elif model_class == 'lstm' or model_class == 'lstm_angle':
        # add LSTM layer with input shape
        model.add(LSTM(256,return_sequences=True,input_shape=input_shape))
        # add dropout
        model.add(Dropout(0.5))
        #add another LSTM layer
        model.add(LSTM(256))
        # add dropout
        model.add(Dropout(0.5))
        # add output layer with softmax activation
        model.add(Dense(output_dim, activation='softmax'))

        # Define adam optimizer as example
        optimizer = optimizers.Adam(lr=1e-4)


# Changmin Seo, Ke Yin

    # Load the weights
    model.load_weights(filedir + '/' + filename)

    # Complie the model
    model.compile(optimizer=optimizer,                   loss='categorical_crossentropy',                   metrics=['accuracy'])
    return model


# In[16]:


# Changmin Seo, Ke Yin

"""Run the read_json_data function with the data json file"""
shuffle = True
use_angle = True
(X_train, y_train, X_validation, y_validation, X_test, y_test) = read_json_data('data_15sec.json', shuffle, use_angle)


# In[17]:


# Changmin Seo

"""Perform data preprocessing"""
(X_train, X_train_mlp, y_train, X_validation, X_validation_mlp, y_validation, X_test, X_test_mlp, y_test, input_dim, output_dim, input_shape) = data_preprocess(X_train, y_train, X_validation, y_validation, X_test, y_test)


# In[18]:


# Changmin Seo

"""Train the mlp_angle_model and evaluate the model"""
model_class = 'mlp_angle'
mlp_angle_model = get_model(model_class)
train_model(model_class, mlp_angle_model, batch_size=20, epochs=10)
evaluate_model(model_class, mlp_angle_model)


# In[19]:


# Changmin Seo

"""evaluate the mlp_angle_model"""
model_class = 'mlp_angle'
evaluate_model(model_class, mlp_angle_model)


# In[20]:


# Changmin Seo

"""Load the mlp_angle_model and evaluate the model"""
model_class = 'mlp_angle'
mlp_angle_model = load_model(model_class)
evaluate_model(model_class, mlp_angle_model)


# In[21]:


# Changmin Seo,ky2327

"""Train the lstm_angle_model and evaluate the model"""
model_class = 'lstm_angle'
lstm_angle_model = get_model(model_class)
train_model(model_class, lstm_angle_model, batch_size=256, epochs=10)
evaluate_model(model_class, lstm_angle_model)


# In[22]:


# Changmin Seo,ky2327

"""evaluate the lstm_angle_model"""
model_class = 'lstm_angle'
evaluate_model(model_class, lstm_angle_model)


# In[23]:


# Changmin Seo,ky2327

"""Load the lstm_angle_model and evaluate the model"""
model_class = 'lstm_angle'
lstm_angle_model = load_model(model_class)
evaluate_model(model_class, lstm_angle_model)


# In[7]:


# Changmin Seo, Ke Yin

"""Run the read_json_data function with the data json file"""
shuffle = True
use_angle = False
(X_train, y_train, X_validation, y_validation, X_test, y_test) = read_json_data('data_15sec.json', shuffle, use_angle)


# In[8]:


# Changmin Seo

"""Perform data preprocessing"""
(X_train, X_train_mlp, y_train, X_validation, X_validation_mlp, y_validation, X_test, X_test_mlp, y_test, input_dim, output_dim, input_shape) = data_preprocess(X_train, y_train, X_validation, y_validation, X_test, y_test)


# In[10]:


# Changmin Seo

"""Train the mlp_model and evaluate the model"""
model_class = 'mlp'
mlp_model = get_model(model_class)
train_model(model_class, mlp_model, batch_size=50, epochs=10)
evaluate_model(model_class, mlp_model)


# In[11]:


# Changmin Seo

"""evaluate the mlp_model"""
model_class = 'mlp'
evaluate_model(model_class, mlp_model)


# In[12]:


# Changmin Seo

"""Load the mlp_model and evaluate the model"""
model_class = 'mlp'
mlp_model = load_model(model_class)
evaluate_model(model_class, mlp_model)


# In[13]:


#ky2327

"""Train the lstm_model and evaluate the model"""
model_class = 'lstm'
lstm_model = get_model(model_class)
train_model(model_class, lstm_model, batch_size=256, epochs=10)
evaluate_model(model_class, lstm_model)


# In[14]:


#ky2327

"""evaluate the lstm_model"""
model_class = 'lstm'
evaluate_model(model_class, lstm_model)


# In[15]:


#ky2327

"""Load the lstm_model and evaluate the model"""
model_class = 'lstm'
lstm_model = load_model(model_class)
evaluate_model(model_class, lstm_model)


# # Annotation Part

# In[53]:


import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

# Changmin Seo

def annotate_result_on_image(model_class, model, X_test, input_path):
    """This function is for annotating prediction results on images"""
    prediction = model.predict(X_test, verbose=0)

    print(prediction.shape)

    for i in range(89):
        # Opening the file
        imageFile = input_path + "/image_" + str(i+1).zfill(4) + ".jpg"
        im1=Image.open(imageFile)

        # Drawing the text on the picture
        draw = ImageDraw.Draw(im1)
        text = "label= input is less than 90 frames"
        draw.text((0, 0),text,(255,255,0))
        draw = ImageDraw.Draw(im1)

        # Save the image with a new name
        im1.save("predict_out/image_" + str(i+1).zfill(4) + ".jpg")

    for i in range(len(prediction)-89):
        # Opening the file
        imageFile = input_path + "/image_" + str(i+90).zfill(4) + ".jpg"
        im1=Image.open(imageFile)

        # Drawing the text on the picture
        draw = ImageDraw.Draw(im1)
        #print(prediction[i])
        text = "label= " + str(np.argmax(prediction[i]))
        draw.text((0, 0),text,(255,255,0))
        draw = ImageDraw.Draw(im1)

        # Save the image with a new name
        im1.save("predict_out/image_" + str(i+90).zfill(4) + ".jpg")


# In[49]:


import json
from pprint import pprint
import numpy as np

# Changmin Seo, Ke Yin

def read_json_data(filename, shuffle, use_angle):
    """This function is for reading json data file which contains pose estimation output"""
    data = json.load(open(filename))

    xy_idx = np.ones(54, dtype=bool);
    for i in range(54):
        if i%3 == 2:
            xy_idx[i] = 0
    x_idx = np.zeros(54, dtype=bool);
    for i in range(54):
        if i%3 == 0:
            x_idx[i] = 1
    y_idx = np.zeros(54, dtype=bool);
    for i in range(54):
        if i%3 == 1:
            y_idx[i] = 1

    frame_per_step = 90

    count = 0

    X_list = []
    y_list = []
    X_val_list = []
    y_val_list = []
    for key in data.keys():
        for key2 in data[key].keys():
            count = count + 1
            # load a specific data
            if key != 'W3' or key2 != '15':
                continue
            print(key, key2, 'count= ', count)
            if key2 == '1R' or key2 == '2R' or key2 == '3R' or key2 == '4R':
                right_flag = True
            else:
                right_flag = False
            if key2 == '4R' or key2 == '4L':
                val_data_flag = True
            else:
                val_data_flag = False
            for key3 in data[key][key2].keys():
                if key3 == 'poses':
                    arr = np.array(data[key][key2][key3])
                    if arr.shape[0] != 0:
                        if right_flag:
                            arr[:,x_idx] *= -1
                            arr[:,x_idx] +=270
                        # normalize
                        x_means = np.mean(arr[:,x_idx], axis = 0)
                        y_means = np.mean(arr[:,y_idx], axis = 0)
                        x_mean_video = np.mean(x_means)
                        y_mean_video = np.mean(y_means)
                        x_std_video = np.std(x_means)
                        y_std_video = np.std(y_means)
                        arr[:,x_idx] -= x_mean_video
                        arr[:,y_idx] -= y_mean_video
                        arr[:,x_idx] /= x_std_video
                        arr[:,y_idx] /= y_std_video

                        for i in range(arr.shape[0] - frame_per_step + 1):
                            X_list.append(arr[i:i+frame_per_step,xy_idx])
                            y_list.append(data[key][key2]['label'])

    # list to np array
    X_data = np.array(X_list)
    y_data = np.array(y_list)
    X_val_data = np.array(X_val_list)
    y_val_data = np.array(y_val_list)
    print(X_data.shape)
    print(y_data.shape)
    print(X_val_data.shape)
    print(y_val_data.shape)
    print('Numpy array data generated.')

    # Shuffle data
    if (shuffle == True):
        print('Now shuffle the data.')
        p = np.random.permutation(len(X_data))
        X_data = X_data[p]
        y_data = y_data[p]
        p = np.random.permutation(len(X_val_data))
        X_val_data = X_val_data[p]
        y_val_data = y_val_data[p]
        print('Shuffle finished.')

    print(X_data.shape)
    print(y_data.shape)
    print(X_val_data.shape)
    print(y_val_data.shape)

    return X_data, y_data, X_val_data, y_val_data


# In[50]:


# Changmin Seo, Ke Yin

"""Run the read_json_data function with the data json file"""
shuffle = False
use_angle = False
(X_data, y_data, X_val_data, y_val_data) = read_json_data('data_15sec.json', shuffle, use_angle)


# In[51]:


# Changmin Seo

"""Seperate train and validation data"""
X_train = X_data[:]
y_train = y_data[:]
X_validation = X_val_data[:]
y_validation = y_val_data[:]
X_test = X_val_data[:]
y_test = y_val_data[:]

# Changmin Seo

"""Define input and output dimension for mlp"""
input_dim = X_train.shape[1] * X_train.shape[2]
output_dim = 4

# Changmin Seo

"""Reshape X data to fit as the input_dim for mlp model"""
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_validation = X_validation.astype('float32')
X_train_mlp = X_train.reshape(X_train.shape[0], input_dim)
X_validation_mlp = X_validation.reshape(X_validation.shape[0], input_dim)
X_test_mlp = X_test.reshape(X_test.shape[0], input_dim)
X_test_mlp = X_train_mlp

# Changmin Seo, Ke Yin

"""Change the label to one-hot shape"""
y_train = np_utils.to_categorical(y_train, output_dim)
y_test = np_utils.to_categorical(y_test, output_dim)
y_validation = np_utils.to_categorical(y_validation, output_dim)
y_test = y_train


# In[54]:


# Changmin Seo

"""Load the mlp_model and annotate_result_on_image"""
model_class = 'mlp'
mlp_model = load_model(model_class)
evaluate_model(model_class, mlp_model)
annotate_result_on_image(model_class, mlp_model, X_test_mlp, 'images_15sec/W3_15L')


# In[28]:


# Changmin Seo

"""Check the ratio of the result label"""
import collections
prediction = mlp_model.predict(X_test_mlp, verbose=0)
print(prediction.shape)
print(np.argmax(prediction, axis=1))
print(collections.Counter(np.argmax(prediction, axis=1)))
