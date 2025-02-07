
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from tensorflow import keras

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



def preprocess_image(image, label, nclasses=10):
    #image = tf.cast(image, tf.float32) / 255.
    image = (tf.cast(image, tf.float32) - 127.5 ) / 127.5
    label = tf.one_hot(tf.squeeze(label), nclasses)
    return image, label


def load_image_train_data (dataset_name):

    if (dataset_name == 'svhn' ):
        dataset_name = 'svhn_cropped'

    X_train, Y_train = tfds.as_numpy(tfds.load(dataset_name, split='train', shuffle_files=True, data_dir='./datasets/', batch_size = -1, as_supervised=True))
    X_test, Y_test = tfds.as_numpy(tfds.load(dataset_name,split='test', shuffle_files=True, data_dir='./datasets/', batch_size=-1,as_supervised=True))

    _, info = tfds.load(dataset_name, split='train', with_info=True, as_supervised=True)

    n_classes   = info.features['label'].num_classes  

    X_train, Y_train    = preprocess_image(X_train, Y_train, n_classes)
    X_test, Y_test      = preprocess_image(X_test, Y_test, n_classes)

    return X_train, Y_train, X_test, Y_test

def get_image_info(dataset_name):

    if (dataset_name == 'svhn' ):
        dataset_name = 'svhn_cropped'

    _, info = tfds.load(dataset_name, split='train', with_info=True, as_supervised=True)
    #print("info: ", info)
    input_shape = info.features['image'].shape 
    n_classes   = info.features['label'].num_classes  
    return input_shape, n_classes


def preprocess_svhn(image, label, nclasses=10):
    image = tf.cast(image, tf.float32) / 255.
    label = tf.one_hot(tf.squeeze(label), nclasses)
    return image, label

def load_train_data_svhn ():

    X_train, Y_train = tfds.as_numpy(tfds.load('svhn_cropped', split='train', shuffle_files=True, data_dir='./datasets/', batch_size = -1, as_supervised=True))
    
    X_test, Y_test = tfds.as_numpy(tfds.load('svhn_cropped',split='test', shuffle_files=True, data_dir='./datasets/', batch_size=-1,as_supervised=True))

    X_train, Y_train    = preprocess_svhn(X_train, Y_train)
    X_test, Y_test      = preprocess_svhn(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def get_info_svhn():
    _, info = tfds.load('svhn_cropped', split='train', with_info=True, as_supervised=True)
    #print("info: ", info)
    input_shape = info.features['image'].shape 
    n_classes   = info.features['label'].num_classes  
    return input_shape, n_classes


def preprocess_cifar10(image, label, nclasses=10):
    image = (tf.cast(image, tf.float32) - 127.5 ) / 127.5
    label = tf.one_hot(tf.squeeze(label), nclasses)
    return image, label

def load_train_data_cifar10 ():

    X_train, Y_train = tfds.as_numpy(tfds.load('cifar10', split='train', shuffle_files=True, data_dir='./datasets/', batch_size = -1, as_supervised=True))
    X_test, Y_test = tfds.as_numpy(tfds.load('cifar10',split='test', shuffle_files=True, data_dir='./datasets/', batch_size=-1,as_supervised=True))

    _, info = tfds.load('cifar10', split='train', with_info=True, as_supervised=True)
    n_classes   = info.features['label'].num_classes  

    X_train_mean  = np.mean((X_train), axis=0)
    X_train = X_train - X_train_mean
    X_test  = X_test - X_train_mean

    X_train = tf.cast(X_train, tf.float32) / 127.5
    Y_train = tf.one_hot(tf.squeeze(Y_train), n_classes)

    X_test = tf.cast(X_test, tf.float32) / 127.5
    Y_test = tf.one_hot(tf.squeeze(Y_test), n_classes)

    return X_train, Y_train, X_test, Y_test

def get_info_cifar10():
    _, info = tfds.load('cifar10', split='train', with_info=True, as_supervised=True)
    #print("info: ", info)
    input_shape = info.features['image'].shape 
    n_classes   = info.features['label'].num_classes  
    return input_shape, n_classes


def load_train_data_jets_hlf ():

    data = fetch_openml('hls4ml_lhc_jets_hlf', data_home ='./datasets')
    x, y = data['data'], data['target']
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, 5)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test

