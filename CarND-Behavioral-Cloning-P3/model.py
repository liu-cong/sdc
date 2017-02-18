import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam

import utils
 
############
# load csv data
############
csv_path = 'data/driving_log.csv'
data = pd.read_csv(csv_path)
data[['steering']] = data[['steering']].apply(pd.to_numeric)
data= data[data['speed']>3].reset_index(drop=True)#remove data with speed<3

# create two generators for training and validation
train_batch_size = 128
validation_batch_size = 32
train_generator = utils.batch_generate_images(data,utils.preprocess_image_file_train, train_batch_size)
valid_generator = utils.batch_generate_images(data, utils.preprocess_image_file_predict, validation_batch_size)

############
# Train the model
############
tf.python.control_flow_ops = tf
fileJson = 'model_nvidia7' + '.json'
fileH5 = 'model_nvidia7' + '.h5'
number_of_epochs = 20
number_of_samples_per_epoch = train_batch_size*150
number_of_validation_samples = validation_batch_size*100
learning_rate = 1e-4
model=utils.get_model()    
model.compile(optimizer=Adam(learning_rate), loss="mse", )
try: #try to load the weights if previously saved
    model.load_weights(fileH5)
    print('Resume training from previously saved weights.')
except:
    print('Training from scratch.')
    pass
history = model.fit_generator(train_generator,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=valid_generator,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

utils.save_model(fileJson, fileH5)