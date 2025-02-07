from functools import wraps
from tensorflow.keras.models import Model, clone_model
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D

import time
# import keras_tuner as kt
from sklearn.model_selection import KFold
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow.keras.regularizers import l1
from qkeras.utils import _add_supported_quantized_objects
from .utility import *

import numpy as np

import pdb

def separable(X_train, Y_train, X_val, Y_val, loss, optimizer, metric = ["accuracy"], callbacks = [], n_epochs = 30, max_trials = 10):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = func(*args, **kwargs)
            def build_model(hp):
                origin_conv_layers = []
                conv_layers = []
                def find_conv_layers(layers):
                    for layer in layers:
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            origin_conv_layers.append(layer.name)

                def _clone_fn(layer):
                    if isinstance(layer, tf.keras.layers.Conv2D) and layer.name in conv_layers:
                        config = layer.get_config()
                        return tf.keras.layers.SeparableConv2D(name = config['name'], filters = config['filters'],
                                                               kernel_size = config["kernel_size"],
                                                               strides=config['strides'],
                                                               padding=config['padding'],
                                                               data_format=config['data_format'],
                                                               dilation_rate=config['dilation_rate'],
                                                               depth_multiplier=1,
                                                               activation=config['activation'],
                                                               kernel_initializer='lecun_uniform',
                                                               kernel_regularizer=l1(0.0001),
                                                               use_bias=config['use_bias'])
                    return layer

                find_conv_layers(model.layers)
                if not origin_conv_layers:
                    return model
                conv_layers = [origin_conv_layers[-1]]
                conv_layers = conv_layers + [layer for layer in origin_conv_layers[:-1] if hp.Boolean(layer)]
                new_model = clone_model(model, clone_function=_clone_fn)
                new_model.compile(loss=loss, optimizer=optimizer, metrics=metric)
                return new_model

            tuner = kt.BayesianOptimization(
                hypermodel=build_model,
                objective="val_loss",
                max_trials=max_trials,
                num_initial_points=5,
                tune_new_entries=True,
                allow_new_entries=True,
                overwrite = True,
                **kwargs)

            tuner.search_space_summary()
            tuner.search(X_train, Y_train,
                            epochs = n_epochs,
                            validation_data = (X_val, Y_val),
                            callbacks = callbacks)
            best_model = tuner.get_best_models(1)[0]

            return best_model
        return wrapper
    return inner

def scale(X_train, Y_train, X_val, Y_val, loss, optimizer, metric = ["accuracy"], callbacks = [], n_epochs = 30, max_trials = 10):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = func(*args, **kwargs)
            def build_model(hp):
                origin_layers_name = []
                origin_layers_units = []
                origin_layers_dic={}
                def scan_layers(layers):
                    for layer in layers:
                        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                            config = layer.get_config()
                            origin_layers_name.append(layer.name)
                            origin_layers_units.append(config['filters'] if isinstance(layer, tf.keras.layers.Conv2D) else config['units'])

                def _clone_fn(layer):
                    if layer.name in origin_layers_name:
                        config = layer.get_config()
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            return tf.keras.layers.Conv2D(name = config['name'], filters = origin_layers_dic[config['name']],
                                                                   kernel_size = config["kernel_size"],
                                                                   strides=config['strides'],
                                                                   padding=config['padding'],
                                                                   data_format=config['data_format'],
                                                                   dilation_rate=config['dilation_rate'],
                                                                   activation=config['activation'],
                                                                   kernel_initializer='lecun_uniform',
                                                                   kernel_regularizer=l1(0.0001),
                                                                   use_bias=config['use_bias'])
                        elif isinstance(layer, tf.keras.layers.Dense):
                            return tf.keras.layers.Dense(name = config['name'], units = origin_layers_dic[config['name']], activation = config['activation'],  use_bias=config['use_bias'])
                    return layer

                scan_layers(model.layers)
                if not origin_layers_name:
                    return model
                for name, unit in zip(origin_layers_name,origin_layers_units):
                    unit = hp.Int(name = name, min_value = max(16, unit - 16), max_value = unit + 16, step=8)
                origin_layers_dic = dict(zip(origin_layers_name,origin_layers_units))
                new_model = clone_model(model, clone_function=_clone_fn)
                new_model.compile(loss=loss, optimizer=optimizer, metrics=metric)
                return new_model

            tuner = kt.BayesianOptimization(
                hypermodel=build_model,
                objective="val_loss",
                max_trials=max_trials,
                num_initial_points=5,
                tune_new_entries=True,
                allow_new_entries=True,
                overwrite = True,
                **kwargs)

            tuner.search_space_summary()
            tuner.search(X_train, Y_train,
                            epochs = n_epochs,
                            validation_data = (X_val, Y_val),
                            callbacks = callbacks)
            best_model = tuner.get_best_models(1)[0]

            return best_model
        return wrapper
    return inner

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
# from tf.keras.wrappers.scikit_learn import KerasClassifier
from bayes_opt import BayesianOptimization as BOptimize
import sys

def scale_simple_v1(scale_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            if not scale_cfg['enable']:
                return model

            write_line_to_log(log_cfg, "start scaling")
            write_line_to_log(log_cfg, f"scaling auto: {str(scale_cfg['auto'])}")
            write_line_to_log(log_cfg, f"scaling step: {str(scale_cfg['step'])}")

            accuracy_list = []
            model_list = []
            iter = 0
            scaling_patience = scale_cfg['patience']

            while True:
                scaling_rate = 1 - (1 / scale_cfg['step']) * iter if scale_cfg['auto'] else scale_cfg['rate']
                write_line_to_log(log_cfg, "\n\n")
                write_line_to_log(log_cfg, f"iteration {str(iter)}")
                write_line_to_log(log_cfg, f"scaling rate: {str(scaling_rate)}")
                write_line_to_log(log_cfg, f"scaling patience: {str(scaling_patience)}")

                scaled_model = tf.keras.Sequential()

                # Iterate over the layers of the original model
                for layer in model.layers:
                    # print(layer.name)
                    # Check if the layer is a Conv2D layer
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        # Scale down the filters by a factor of 2
                        new_filters = max(1, int(layer.filters * scaling_rate))

                        # Create a new Conv2D layer with scaled-down filters
                        new_layer = tf.keras.layers.Conv2D(name = layer.name,
                                                        filters = new_filters,
                                                        kernel_size = layer.kernel_size,
                                                        strides = layer.strides,
                                                        activation = layer.activation,
                                                        padding=layer.padding,
                                                        kernel_initializer=layer.kernel_initializer,
                                                        kernel_regularizer = layer.kernel_regularizer,
                                                        use_bias = layer.use_bias
                                                        )
                        # Add the new layer to the new model
                        scaled_model.add(new_layer)

                    elif isinstance(layer, tf.keras.layers.Dense):
                        config = layer.get_config()
                        new_unit = max(1, int(config['units'] * scaling_rate)) if (layer.name!='output_dense') else config['units']

                        # print(f"layer {str(config['name'])} scale from {str(config['units'])} to {str(new_unit)}")
                        new_layer = tf.keras.layers.Dense(name = config['name'],
                                                    units = new_unit,
                                                    activation = config['activation'],
                                                    kernel_initializer=layer.kernel_initializer,
                                                    kernel_regularizer = layer.kernel_regularizer,
                                                    use_bias=config['use_bias'])
                        scaled_model.add(new_layer)

                    elif isinstance(layer, tf.keras.layers.BatchNormalization):
                        scaled_model.add(tf.keras.layers.BatchNormalization())

                    elif isinstance(layer, tf.keras.layers.Activation):
                        scaled_model.add(tf.keras.layers.Activation(activation = layer.activation,
                                                                    name = layer.name))

                    elif isinstance(layer, tf.keras.layers.MaxPool2D):
                        scaled_model.add(tf.keras.layers.MaxPool2D(name = layer.name,
                                                                    pool_size = layer.pool_size))

                    elif isinstance(layer, tf.keras.layers.Flatten):
                        scaled_model.add(tf.keras.layers.Flatten())

                    else:
                        # For non-Conv2D layers, add them to the new model as is
                        scaled_model.add(layer)
                        print(layer.get_config())
                        scaled_model.summary()

                model_list.append(scaled_model)
                model_list[-1].compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                # model_list[-1].summary()

                # train model
                history = model_list[-1].fit(
                                    train_cfg['X_train'],
                                    train_cfg['Y_train'],
                                    batch_size  = train_cfg['batch_size'],
                                    epochs      = train_cfg['n_epochs'],
                                    validation_split= train_cfg['val_split'],
                                    callbacks   = train_cfg['callbacks'])

                # write_line_to_log(log_cfg, "\n\n")
                # write_line_to_log(log_cfg, "training history:")
                # write_dic_to_log (log_cfg, history.history)

                # test model
                evaluate_res = model_list[-1].evaluate(train_cfg['X_test'], train_cfg['Y_test'])
                accuracy_list.append(evaluate_res[1])
                write_line_to_log(log_cfg, f'accuracy list = {accuracy_list}')
                model_list[-1].save(train_cfg['model_loc'] + f'/scaled_model_iter{iter}.tf')

                # static scaling
                if not scale_cfg['auto']:
                    return scaled_model

                # If the accuracy loss exceeds 2%, stop scaling --> remove last model; break
                if iter != 0:
                    if (accuracy_list[0] - accuracy_list[-1]) / accuracy_list[0] > scale_cfg['threshold']:
                        scaling_patience -= 1
                        old_list_num = len(model_list)
                        model_list.pop(-1)
                        new_list_num = len(model_list)
                        assert new_list_num == old_list_num - 1
                        write_line_to_log(log_cfg, "remove the model")

                        if scaling_patience == 0:
                            break
                    else:
                        scaling_patience = scale_cfg['patience']

                # iter++, if iter >= step --> stop scaling
                iter += 1
                if iter == scale_cfg['step']:
                    break

            return model_list[-1]

        return wrapper
    return inner


def scale_simple_v1(scale_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            if not scale_cfg['enable']:
                return model

            write_line_to_log(log_cfg, "start scaling")
            write_line_to_log(log_cfg, f"scaling auto: {str(scale_cfg['auto'])}")
            write_line_to_log(log_cfg, f"scaling step: {str(scale_cfg['step'])}")

            accuracy_list = []
            model_list = []
            iter = 0
            scaling_patience = scale_cfg['patience']

            while True:
                scaling_rate = 1 - (1 / scale_cfg['step']) * iter if scale_cfg['auto'] else scale_cfg['rate']
                write_line_to_log(log_cfg, "\n\n")
                write_line_to_log(log_cfg, f"iteration {str(iter)}")
                write_line_to_log(log_cfg, f"scaling rate: {str(scaling_rate)}")
                write_line_to_log(log_cfg, f"scaling patience: {str(scaling_patience)}")

                scaled_model = tf.keras.Sequential()

                # Iterate over the layers of the original model
                for layer in model.layers:
                    # print(layer.name)
                    # Check if the layer is a Conv2D layer
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        # Scale down the filters by a factor of 2
                        new_filters = max(1, int(layer.filters * scaling_rate))

                        # Create a new Conv2D layer with scaled-down filters
                        new_layer = tf.keras.layers.Conv2D(name = layer.name,
                                                        filters = new_filters,
                                                        kernel_size = layer.kernel_size,
                                                        strides = layer.strides,
                                                        activation = layer.activation,
                                                        padding=layer.padding,
                                                        kernel_initializer=layer.kernel_initializer,
                                                        kernel_regularizer = layer.kernel_regularizer,
                                                        use_bias = layer.use_bias
                                                        )
                        # Add the new layer to the new model
                        scaled_model.add(new_layer)

                    elif isinstance(layer, tf.keras.layers.Dense):
                        config = layer.get_config()
                        new_unit = max(1, int(config['units'] * scaling_rate)) if (layer.name!='output_dense') else config['units']

                        # print(f"layer {str(config['name'])} scale from {str(config['units'])} to {str(new_unit)}")
                        new_layer = tf.keras.layers.Dense(name = config['name'],
                                                    units = new_unit,
                                                    activation = config['activation'],
                                                    kernel_initializer=layer.kernel_initializer,
                                                    kernel_regularizer = layer.kernel_regularizer,
                                                    use_bias=config['use_bias'])
                        scaled_model.add(new_layer)

                    elif isinstance(layer, tf.keras.layers.BatchNormalization):
                        scaled_model.add(tf.keras.layers.BatchNormalization())

                    elif isinstance(layer, tf.keras.layers.Activation):
                        scaled_model.add(tf.keras.layers.Activation(activation = layer.activation,
                                                                    name = layer.name))

                    elif isinstance(layer, tf.keras.layers.MaxPool2D):
                        scaled_model.add(tf.keras.layers.MaxPool2D(name = layer.name,
                                                                    pool_size = layer.pool_size))

                    elif isinstance(layer, tf.keras.layers.Flatten):
                        scaled_model.add(tf.keras.layers.Flatten())

                    else:
                        # For non-Conv2D layers, add them to the new model as is
                        scaled_model.add(layer)
                        print(layer.get_config())
                        scaled_model.summary()

                model_list.append(scaled_model)
                model_list[-1].compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                # model_list[-1].summary()

                # train model
                history = model_list[-1].fit(
                                    train_cfg['X_train'],
                                    train_cfg['Y_train'],
                                    batch_size  = train_cfg['batch_size'],
                                    epochs      = train_cfg['n_epochs'],
                                    validation_split= train_cfg['val_split'],
                                    callbacks   = train_cfg['callbacks'])

                # write_line_to_log(log_cfg, "\n\n")
                # write_line_to_log(log_cfg, "training history:")
                # write_dic_to_log (log_cfg, history.history)

                # test model
                evaluate_res = model_list[-1].evaluate(train_cfg['X_test'], train_cfg['Y_test'])
                accuracy_list.append(evaluate_res[1])
                write_line_to_log(log_cfg, f'accuracy list = {accuracy_list}')
                model_list[-1].save(train_cfg['model_loc'] + f'/scaled_model_iter{iter}.tf')

                # static scaling
                if not scale_cfg['auto']:
                    return scaled_model

                # If the accuracy loss exceeds 2%, stop scaling --> remove last model; break
                if iter != 0:
                    if (accuracy_list[0] - accuracy_list[-1]) / accuracy_list[0] > scale_cfg['threshold']:
                        scaling_patience -= 1
                        model_list.pop(-1)
                        write_line_to_log(log_cfg, "remove the model")

                        if scaling_patience == 0:
                            break
                    else:
                        scaling_patience = scale_cfg['patience']

                # iter++, if iter >= step --> stop scaling
                iter += 1
                if iter == scale_cfg['step']:
                    break

            return model_list[-1]

        return wrapper
    return inner

def scale_auto_v4(scale_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            if not scale_cfg['enable']:
                return model

            if not scale_cfg['auto']:
                write_line_to_log(log_cfg, "start static scaling")
                write_line_to_log(log_cfg, f"scaling rate: {str(scale_cfg['rate'])}")

                layers = model.layers
                accuracy_list = []
                model_list = []
                iter = 0
                while True:
                    scaling_rate = 1 - scale_cfg['rate'] * iter
                    # Create a new model with scaled-down filters
                    scaled_model = tf.keras.Sequential()

                    # Iterate over the layers of the original model
                    for layer in model.layers:
                        # print(layer.name)
                        # Check if the layer is a Conv2D layer
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            # Scale down the filters by a factor of 2
                            new_filters = int(layer.filters * scaling_rate)

                            # Create a new Conv2D layer with scaled-down filters
                            new_layer = tf.keras.layers.Conv2D(name = layer.name,
                                                            filters = new_filters,
                                                            kernel_size = layer.kernel_size,
                                                            strides = layer.strides,
                                                            activation = layer.activation,
                                                            padding=layer.padding,
                                                            kernel_initializer=layer.kernel_initializer,
                                                            kernel_regularizer = layer.kernel_regularizer,
                                                            use_bias = layer.use_bias
                                                            )
                            # Add the new layer to the new model
                            scaled_model.add(new_layer)

                        elif isinstance(layer, tf.keras.layers.Dense):
                            config = layer.get_config()
                            new_unit = int(config['units'] * scaling_rate) if (layer.name!='output_dense') else config['units']

                            # print(f"layer {str(config['name'])} scale from {str(config['units'])} to {str(new_unit)}")
                            new_layer = tf.keras.layers.Dense(name = config['name'],
                                                        units = new_unit,
                                                        activation = config['activation'],
                                                        kernel_initializer=layer.kernel_initializer,
                                                        kernel_regularizer = layer.kernel_regularizer,
                                                        use_bias=config['use_bias'])
                            scaled_model.add(new_layer)

                        elif isinstance(layer, tf.keras.layers.BatchNormalization):
                            scaled_model.add(tf.keras.layers.BatchNormalization())

                        elif isinstance(layer, tf.keras.layers.Activation):
                            scaled_model.add(tf.keras.layers.Activation(activation = layer.activation,
                                                                        name = layer.name))

                        elif isinstance(layer, tf.keras.layers.MaxPool2D):
                            scaled_model.add(tf.keras.layers.MaxPool2D(name = layer.name,
                                                                        pool_size = layer.pool_size))

                        elif isinstance(layer, tf.keras.layers.Flatten):
                            scaled_model.add(tf.keras.layers.Flatten())

                        else:
                            # For non-Conv2D layers, add them to the new model as is
                            scaled_model.add(layer)
                            print(layer.get_config())
                            scaled_model.summary()

                    model_list.append(scaled_model)
                    model_list[-1].compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                    model_list[-1].summary()

                    # train model
                    history = model_list[-1].fit(
                                        train_cfg['X_train'],
                                        train_cfg['Y_train'],
                                        batch_size  = train_cfg['batch_size'],
                                        epochs      = train_cfg['n_epochs'],
                                        validation_split= train_cfg['val_split'],
                                        callbacks   = train_cfg['callbacks'])

                    # write_line_to_log(log_cfg, "\n\n")
                    # write_line_to_log(log_cfg, "training history:")
                    # write_dic_to_log (log_cfg, history.history)

                    # test model
                    evaluate_res = model_list[-1].evaluate(train_cfg['X_test'], train_cfg['Y_test'])
                    accuracy_list.append(evaluate_res[1])
                    write_line_to_log(log_cfg, f'accuracy list = {accuracy_list}')

                    # If the accuracy loss exceeds 2%, stop scaling --> remove last model; break
                    if iter != 0:
                        if (accuracy_list[0] - accuracy_list[-1]) / accuracy_list[0] > scale_cfg['threshold']:
                            model_list.pop(-1)
                            break

                    # iter++, if iter >= step --> stop scaling
                    iter += 1
                    if iter >= 1 / scale_cfg['rate']:
                        break

                print("model after static scaling")
                return model_list[-1]

            write_line_to_log(log_cfg, "start auto scaling")

            # # get original model accuracy
            # model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # # train model
            # model.fit(  train_cfg['X_train'],
            #             train_cfg['Y_train'],
            #             batch_size  = train_cfg['batch_size'],
            #             epochs      = train_cfg['n_epochs'],
            #             validation_split= train_cfg['val_split'],
            #             callbacks   = train_cfg['callbacks'])

            # # test model
            # orgn_acc = model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])

            # write_line_to_log(log_cfg, "\n\n")
            # write_line_to_log(log_cfg, f"original model accuracy: {str(orgn_acc)}")

            def build_model(hp):
                origin_layers_name = []
                origin_layers_units = []
                origin_layers_dic={}
                def scan_layers(layers):
                    for layer in layers:
                        config2 = layer.get_config()
                        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                            config = layer.get_config()
                            origin_layers_name.append(layer.name)
                            origin_layers_units.append(config['filters'] if isinstance(layer, tf.keras.layers.Conv2D) else config['units'])

                def _clone_fn(layer):
                    config = layer.get_config()
                    if layer.name in origin_layers_name:
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            #print(config['name'])
                            #print(origin_layers_dic[config['name']])
                            return tf.keras.layers.Conv2D(
                                            name = config['name'],
                                            filters = int(origin_layers_dic[config['name']]),
                                            kernel_size = config["kernel_size"],
                                            strides=config['strides'],
                                            padding=config['padding'],
                                            data_format=config['data_format'],
                                            dilation_rate=config['dilation_rate'],
                                            activation=config['activation'],
                                            kernel_initializer='lecun_uniform',
                                            kernel_regularizer=l1(0.0001),
                                            use_bias=config['use_bias'])
                        elif isinstance(layer, tf.keras.layers.Dense):
                            return tf.keras.layers.Dense(
                                            name = config['name'],
                                            units = int(origin_layers_dic[config['name']]),
                                            activation = config['activation'],
                                            use_bias=config['use_bias'])
                    # BN needs reset
                    #if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Add) :
                    else:
                        #config = layers[i].get_config()
                        #print('Walkie')
                        #print(config)
                        layer = layer.from_config( config )
                        return layer
                    return layer

                scan_layers(model.layers)
                #exit()
                if not origin_layers_name:
                    print(origin_layers_name)
                    print("return original model")
                    return model
                origin_layers_dic = dict(zip(origin_layers_name,origin_layers_units))
                for name in origin_layers_dic:
                    if(name!='output_dense'):
                        unit = origin_layers_dic[name]
                        origin_layers_dic[name] = hp.Float(name = name,
                                                           min_value = 0,
                                                           max_value = unit,
                                                           step = unit * scale_cfg['rate'])
                print(origin_layers_dic)
                new_model = tf.keras.models.clone_model(model, clone_function=_clone_fn)
                new_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                return new_model

            # Define the objective function with constraint on maximum loss accuracy
            def model_params_size(hp):
                model = build_model(hp)
                model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
                model.fit(train_cfg['X_train'], train_cfg['Y_train'], batch_size  = train_cfg['batch_size'],
                          epochs = train_cfg['n_epochs'], validation_split= train_cfg['val_split'], callbacks   = train_cfg['callbacks'])

                # Evaluate the model on the test set
                # test_loss, test_accuracy = model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])

                # if (orgn_acc - test_accuracy) / orgn_acc > scale_cfg['threshold']:
                #     return -np.inf  # Return a large negative value if accuracy is below the threshold

                return -model.count_params()  # Return negative model size as the objective

            # Perform Bayesian optimization
            tuner = kt.BayesianOptimization(hypermodel=build_model,
                                            objective=kt.Objective('val_model_params_size', direction='max'),
                                            max_trials=scale_cfg['max_trials'])
            tuner.search_space_summary()
            tuner.search(train_cfg['X_train'],
                         train_cfg['Y_train'],
                         batch_size  = train_cfg['batch_size'],
                         epochs      = train_cfg['n_epochs'],
                         validation_split= train_cfg['val_split'],
                         callbacks   = train_cfg['callbacks'])

            tuner.results_summary()

            # Get the best hyperparameters
            best_hyperparameters = tuner.get_best_hyperparameters()[0]

            # Build the final model with the best hyperparameters
            final_model = tuner.hypermodel.build(best_hyperparameters)
            final_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
            final_model.fit(train_cfg['X_train'], train_cfg['Y_train'], batch_size  = train_cfg['batch_size'],
                            epochs = train_cfg['n_epochs'], validation_split= train_cfg['val_split'], callbacks = train_cfg['callbacks'])


            print("best model from keras tuner get_best_model")
            final_model.summary()

            return final_model
        return wrapper
    return inner

def scale_auto_v3(scale_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            write_line_to_log(log_cfg, "start auto scaling")
            def build_model(hp):
                origin_layers_name = []
                origin_layers_units = []
                origin_layers_dic={}
                def scan_layers(layers):
                    for layer in layers:
                        config2 = layer.get_config()
                        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                            config = layer.get_config()
                            origin_layers_name.append(layer.name)
                            origin_layers_units.append(config['filters'] if isinstance(layer, tf.keras.layers.Conv2D) else config['units'])

                def _clone_fn(layer):
                    config = layer.get_config()
                    if layer.name in origin_layers_name:
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            #print(config['name'])
                            #print(origin_layers_dic[config['name']])
                            return tf.keras.layers.Conv2D(
                                            name = config['name'],
                                            filters = origin_layers_dic[config['name']],
                                            kernel_size = config["kernel_size"],
                                            strides=config['strides'],
                                            padding=config['padding'],
                                            data_format=config['data_format'],
                                            dilation_rate=config['dilation_rate'],
                                            activation=config['activation'],
                                            kernel_initializer='lecun_uniform',
                                            kernel_regularizer=l1(0.0001),
                                            use_bias=config['use_bias'])
                        elif isinstance(layer, tf.keras.layers.Dense):
                            return tf.keras.layers.Dense(
                                            name = config['name'],
                                            units = origin_layers_dic[config['name']],
                                            activation = config['activation'],
                                            use_bias=config['use_bias'])
                    # BN needs reset
                    #if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Add) :
                    else:
                        #config = layers[i].get_config()
                        #print('Walkie')
                        #print(config)
                        layer = layer.from_config( config )
                        return layer
                    return layer

                scan_layers(model.layers)
                #exit()
                if not origin_layers_name:
                    print(origin_layers_name)
                    print("return original model")
                    return model
                origin_layers_dic = dict(zip(origin_layers_name,origin_layers_units))
                for name in origin_layers_dic:
                    if(name!='output_dense'):
                        unit = origin_layers_dic[name]
                        origin_layers_dic[name] = hp.Int(name = name, min_value = max(16, unit - 16), max_value = unit + 16, step=8)
                print(origin_layers_dic)
                new_model = tf.keras.models.clone_model(model, clone_function=_clone_fn)
                new_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                #layers = [l for l in new_model.layers]
                #x = layers[0].output
                #for i in range(1, len(layers)):
                #    x = layers[i](x)
                #new_model_rebulit = Model(inputs=[layers[0].input], outputs=[x])
                ##new_model_rebulit.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
                #new_model_rebulit.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                #new_model_rebulit.summary()
                #del new_model
                #return new_model_rebulit
                return new_model

            tuner = kt.BayesianOptimization(
                hypermodel=build_model,
                objective="val_loss",
                max_trials=scale_cfg['max_trials'],
                num_initial_points=5,
                tune_new_entries=True,
                allow_new_entries=True,
                overwrite = True,
                **kwargs)

            tuner.search_space_summary()
            tuner.search(train_cfg['X_train'],
                         train_cfg['Y_train'],
                         batch_size  = train_cfg['batch_size'],
                         epochs      = train_cfg['n_epochs'],
                         validation_split= train_cfg['val_split'],
                         callbacks   = train_cfg['callbacks'])

            tuner.results_summary()
            best_tuner_model = tuner.get_best_models(1)[0]
            #best_model = models[0]
            print("best model from keras tuner get_best_model")
            best_tuner_model.summary()

            # rebuilt the model
            best_tuner_model.save_weights(train_cfg['model_weights_loc'])
            layers = [l for l in best_tuner_model.layers]
            x = layers[0].output
            for i in range(1, len(layers)):
                x = layers[i](x)

            new_best_model = Model(inputs=[layers[0].input], outputs=[x])
            new_best_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
            #new_best_model.summary()
            new_best_model.load_weights(train_cfg['model_weights_loc'])

            #new_best_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
            #print("\nnew best model")
            #new_best_model.summary()
            #new_best_model.load_weights(train_cfg['model_weights_loc'])

            #best_hps = tuner.get_best_hyperparameters()[0]
            #best_hps_model = build_model(best_hps)
            #best_hps_model.summary()
            #history = best_hps_model.fit(train_cfg['X_train'],
            #             train_cfg['Y_train'],
            #             batch_size  = train_cfg['batch_size'],
            #             epochs      = train_cfg['n_epochs'],
            #             validation_split= train_cfg['val_split'],
            #             callbacks   = train_cfg['callbacks'])

            #h_model = model
            #best_model = h_model.build(best_hps[0])
            #history = best_model.fit(train_cfg['X_train'],
            #             train_cfg['Y_train'],
            #             batch_size  = train_cfg['batch_size'],
            #             epochs      = train_cfg['n_epochs'],
            #             validation_split= train_cfg['val_split'],
            #             callbacks   = train_cfg['callbacks'])
            #
            #write_line_to_log(log_cfg, "\n\n")
            #write_line_to_log(log_cfg, "final history:")
            #write_dic_to_log (log_cfg, history.history)


            return new_best_model
            #return best_tuner_model
        return wrapper
    return inner

def scale_auto_v2(scale_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            write_line_to_log(log_cfg, "start auto scaling")
            def build_model(hp):
                origin_layers_name = []
                origin_layers_units = []
                origin_layers_dic={}
                def scan_layers(layers):
                    for layer in layers:
                        config2 = layer.get_config()
                        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                            config = layer.get_config()
                            origin_layers_name.append(layer.name)
                            origin_layers_units.append(config['filters'] if isinstance(layer, tf.keras.layers.Conv2D) else config['units'])


                scan_layers(model.layers)
                if not origin_layers_name:
                    print(origin_layers_name)
                    print("return original model")
                    return model
                origin_layers_dic = dict(zip(origin_layers_name,origin_layers_units))
                for name in origin_layers_dic:
                    if(name!='output_dense'):
                        unit = origin_layers_dic[name]
                        origin_layers_dic[name] = hp.Int(name = name, min_value = max(16, unit - 16), max_value = unit + 16, step=8)
                print(origin_layers_dic)

                #new_model = clone_model(model, clone_function=_clone_fn)

                layers = [l for l in model.layers]
                x = layers[0].output
                for i in range(1, len(layers)):
                    if isinstance(layers[i], tf.keras.layers.Conv2D):
                        config = layers[i].get_config()
                        #print(config)
                        layer_name = config['name']
                        print(f'{layer_name} filters', config['filters'])
                        config['filters'] = origin_layers_dic[config['name']]
                        layers[i] = layers[i].from_config( config )

                        config = layers[i].get_config()
                        print(f'{layer_name} new filters', config['filters'])
                    elif isinstance(layers[i], tf.keras.layers.Dense):
                        config = layers[i].get_config()
                        #print(config)
                        layer_name = config['name']
                        print('{layer_name} units', config['units'])
                        config['units'] = origin_layers_dic[config['name']]
                        layers[i] = layers[i].from_config( config )

                        config = layers[i].get_config()
                        #print(config)
                        print('{layer_name} new units', config['units'])
                    elif isinstance(layers[i], tf.keras.layers.BatchNormalization):
                        config = layers[i].get_config()
                        print(config)
                        layers[i] = layers[i].from_config( config )

                    x = layers[i](x)
                new_model = Model(inputs=[layers[0].input], outputs=[x])

                new_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                new_model.summary()
                print(origin_layers_dic)

                return new_model

            tuner = kt.BayesianOptimization(
                hypermodel=build_model,
                objective="val_loss",
                max_trials=scale_cfg['max_trials'],
                num_initial_points=5,
                tune_new_entries=True,
                allow_new_entries=True,
                overwrite = True,
                **kwargs)

            tuner.search_space_summary()
            tuner.search(train_cfg['X_train'],
                         train_cfg['Y_train'],
                         batch_size  = train_cfg['batch_size'],
                         epochs      = train_cfg['n_epochs'],
                         validation_split= train_cfg['val_split'],
                         callbacks   = train_cfg['callbacks'])

            tuner.results_summary()
            best_tuner_model = tuner.get_best_models(1)[0]
            #best_model = models[0]
            print("best model from keras tuner get_best_model")
            best_tuner_model.summary()

            # rebuilt the model
            best_tuner_model.save_weights(train_cfg['model_weights_loc'])
            layers = [l for l in best_tuner_model.layers]
            x = layers[0].output
            for i in range(1, len(layers)):
                x = layers[i](x)

            new_best_model = Model(inputs=[layers[0].input], outputs=[x])
            new_best_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
            #new_best_model.summary()
            new_best_model.load_weights(train_cfg['model_weights_loc'])

            #new_best_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
            #print("\nnew best model")
            #new_best_model.summary()
            #new_best_model.load_weights(train_cfg['model_weights_loc'])

            #best_hps = tuner.get_best_hyperparameters()[0]
            #best_hps_model = build_model(best_hps)
            #best_hps_model.summary()
            #history = best_hps_model.fit(train_cfg['X_train'],
            #             train_cfg['Y_train'],
            #             batch_size  = train_cfg['batch_size'],
            #             epochs      = train_cfg['n_epochs'],
            #             validation_split= train_cfg['val_split'],
            #             callbacks   = train_cfg['callbacks'])

            #h_model = model
            #best_model = h_model.build(best_hps[0])
            #history = best_model.fit(train_cfg['X_train'],
            #             train_cfg['Y_train'],
            #             batch_size  = train_cfg['batch_size'],
            #             epochs      = train_cfg['n_epochs'],
            #             validation_split= train_cfg['val_split'],
            #             callbacks   = train_cfg['callbacks'])
            #
            #write_line_to_log(log_cfg, "\n\n")
            #write_line_to_log(log_cfg, "final history:")
            #write_dic_to_log (log_cfg, history.history)


            return new_best_model
            #return best_tuner_model
        return wrapper
    return inner

def scale_auto(scale_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            write_line_to_log(log_cfg, "start auto scaling")
            def build_model(hp):
                origin_layers_name = []
                origin_layers_units = []
                origin_layers_dic={}
                def scan_layers(layers):
                    for layer in layers:
                        config2 = layer.get_config()
                        #print(config2)
                        #print(str(layer.input))
                        #print(str(layer.output))
                        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                            config = layer.get_config()
                            #print(config)
                            #print(str(config))
                            origin_layers_name.append(layer.name)
                            origin_layers_units.append(config['filters'] if isinstance(layer, tf.keras.layers.Conv2D) else config['units'])

                def _clone_fn(layer):
                    if layer.name in origin_layers_name:
                        config = layer.get_config()
                        if isinstance(layer, tf.keras.layers.Conv2D):

                            #print(config['name'])
                            #print(origin_layers_dic[config['name']])
                            return tf.keras.layers.Conv2D(
                                            name = config['name'],
                                            filters = origin_layers_dic[config['name']],
                                            kernel_size = config["kernel_size"],
                                            strides=config['strides'],
                                            padding=config['padding'],
                                            data_format=config['data_format'],
                                            dilation_rate=config['dilation_rate'],
                                            activation=config['activation'],
                                            kernel_initializer='lecun_uniform',
                                            kernel_regularizer=l1(0.0001),
                                            use_bias=config['use_bias'])
                        elif isinstance(layer, tf.keras.layers.Dense):
                            return tf.keras.layers.Dense(
                                            name = config['name'],
                                            units = origin_layers_dic[config['name']],
                                            activation = config['activation'],
                                            use_bias=config['use_bias'])
                    return layer

                scan_layers(model.layers)
                #exit()
                if not origin_layers_name:
                    print(origin_layers_name)
                    print("return original model")
                    return model
                origin_layers_dic = dict(zip(origin_layers_name,origin_layers_units))
                for name in origin_layers_dic:
                    if(name!='output_dense'):
                        unit = origin_layers_dic[name]
                        origin_layers_dic[name] = hp.Int(name = name, min_value = max(16, unit - 16), max_value = unit + 16, step=8)
                print(origin_layers_dic)
                new_model = clone_model(model, clone_function=_clone_fn)
                new_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                #layers = [l for l in new_model.layers]
                #x = layers[0].output
                #for i in range(1, len(layers)):
                #    x = layers[i](x)
                #new_model_rebulit = Model(inputs=[layers[0].input], outputs=[x])
                ##new_model_rebulit.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
                #new_model_rebulit.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                #new_model_rebulit.summary()
                #del new_model
                #return new_model_rebulit
                return new_model

            tuner = kt.BayesianOptimization(
                hypermodel=build_model,
                objective="val_loss",
                max_trials=scale_cfg['max_trials'],
                num_initial_points=5,
                tune_new_entries=True,
                allow_new_entries=True,
                overwrite = True,
                **kwargs)

            tuner.search_space_summary()
            tuner.search(train_cfg['X_train'],
                         train_cfg['Y_train'],
                         batch_size  = train_cfg['batch_size'],
                         epochs      = train_cfg['n_epochs'],
                         validation_split= train_cfg['val_split'],
                         callbacks   = train_cfg['callbacks'])

            tuner.results_summary()
            best_tuner_model = tuner.get_best_models(1)[0]
            #best_model = models[0]
            print("best model from keras tuner get_best_model")
            best_tuner_model.summary()

            # rebuilt the model
            best_tuner_model.save_weights(train_cfg['model_weights_loc'])
            layers = [l for l in best_tuner_model.layers]
            x = layers[0].output
            for i in range(1, len(layers)):
                x = layers[i](x)

            new_best_model = Model(inputs=[layers[0].input], outputs=[x])
            new_best_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
            #new_best_model.summary()
            new_best_model.load_weights(train_cfg['model_weights_loc'])

            #new_best_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
            #print("\nnew best model")
            #new_best_model.summary()
            #new_best_model.load_weights(train_cfg['model_weights_loc'])

            #best_hps = tuner.get_best_hyperparameters()[0]
            #best_hps_model = build_model(best_hps)
            #best_hps_model.summary()
            #history = best_hps_model.fit(train_cfg['X_train'],
            #             train_cfg['Y_train'],
            #             batch_size  = train_cfg['batch_size'],
            #             epochs      = train_cfg['n_epochs'],
            #             validation_split= train_cfg['val_split'],
            #             callbacks   = train_cfg['callbacks'])

            #h_model = model
            #best_model = h_model.build(best_hps[0])
            #history = best_model.fit(train_cfg['X_train'],
            #             train_cfg['Y_train'],
            #             batch_size  = train_cfg['batch_size'],
            #             epochs      = train_cfg['n_epochs'],
            #             validation_split= train_cfg['val_split'],
            #             callbacks   = train_cfg['callbacks'])
            #
            #write_line_to_log(log_cfg, "\n\n")
            #write_line_to_log(log_cfg, "final history:")
            #write_dic_to_log (log_cfg, history.history)


            return new_best_model
            #return best_tuner_model
        return wrapper
    return inner

#def scale(X_train, Y_train, X_val, Y_val, loss, optimizer, metric = ["accuracy"], callbacks = [], n_epochs = 30, max_trials = 10):
def pruning_pretrain_auto(pruning_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)
            if not pruning_cfg['enable']:
                return model

            NSTEPS = len(train_cfg['X_train']) // train_cfg['batch_size']

            # print the size
            write_line_to_log(log_cfg, "start auto pruning")
            for layer in model.layers:
                if layer.__class__.__name__ in ['Conv2D', 'Dense']:
                    w = layer.get_weights()[0]
                    layersize = np.prod(w.shape)
                    write_line_to_log(log_cfg, "{}: w shape is {}".format(layer.name,str(w.shape))) # 0 = weights, 1 = biases
                    write_line_to_log(log_cfg, "{}: w size is {}".format(layer.name,layersize)) # 0 = weights, 1 = biases

            pruning_auto_step = 0
            final_model_num = 0
            cur_acc     = []
            #cur_p_rate  = [0.0,pruning_cfg['start_sparsity']]
            if (pruning_cfg['auto']):
                #cur_p_rate  = [pruning_cfg['start_sparsity']]
                cur_p_rate  = [0.0]
            else:
                cur_p_rate  = [pruning_cfg['target_sparsity']]

            acc_delta   = 1.0
            pruning_rate_delta = 1.0
            model_candidate = []
            model_good = []

            #while (acc_delta > -0.02 and pruning_rate_delta > 0.01 ):
            write_line_to_log(log_cfg, f"pruning enable: {str(pruning_cfg['enable'])}")
            write_line_to_log(log_cfg, f"auto pruning: {str(pruning_cfg['auto'])}")
            while True:
                write_line_to_log(log_cfg, f"pruning step: {pruning_auto_step}")
                write_line_to_log(log_cfg, f"current accuracy list: {str(cur_acc)}" )
                write_line_to_log(log_cfg, f"current pruning rate list: {str(cur_p_rate)}" )
                write_line_to_log(log_cfg, "\n" )
                write_line_to_log(log_cfg, f"last acc delta: {acc_delta}")
                write_line_to_log(log_cfg, f"last pruning rate delta: {pruning_rate_delta}")
                write_line_to_log(log_cfg, f"acc delta threshold: {pruning_cfg['acc_delta_ths']}")
                write_line_to_log(log_cfg, f"pruning rate delta threshold: {pruning_cfg['p_rate_delta_ths']}")

                write_line_to_log(log_cfg, "\n" )
                write_line_to_log(log_cfg, f"curent pruning rate: {cur_p_rate[-1]}")

                if pruning_cfg['enable']:
                    def pruneFunction(layer):
                        pruning_params = {'pruning_schedule':
                            sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                     #final_sparsity = pruning_cfg['target_sparsity'],
                                                     final_sparsity = cur_p_rate[-1],
                                                     begin_step = NSTEPS*2,
                                                     end_step = NSTEPS*6,
                                                     frequency = NSTEPS)
                                         }
                        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.SeparableConv2D):
                            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                        if isinstance(layer, tf.keras.layers.Conv1D):
                            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                        #if isinstance(layer, tf.keras.layers.Dense) and layer.name!='output_dense':
                        if isinstance(layer, tf.keras.layers.Dense):
                            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                        #if layer.name == 'output_dense':
                        #    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params_output)
                        return layer

                    #model_pruned = tf.keras.models.clone_model( model, clone_function=pruneFunction)
                    model_candidate.append( tf.keras.models.clone_model( model, clone_function=pruneFunction))
                else: # no pruning
                    model_candidate.append( model )

                #model = model_pruned
                model_candidate[-1].compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])

                if train_cfg['enable']:
                    history = model_candidate[-1].fit(
                                        train_cfg['X_train'],
                                        train_cfg['Y_train'],
                                        batch_size  = train_cfg['batch_size'],
                                        epochs      = train_cfg['n_epochs'],
                                        validation_split= train_cfg['val_split'],
                                        callbacks   = train_cfg['callbacks'])


                    write_line_to_log(log_cfg, "\n\n")
                    write_line_to_log(log_cfg, "training history:")
                    write_dic_to_log (log_cfg, history.history)

                    model_candidate[-1].summary()
                    bayes_iter = train_cfg['bayes_iter']
                    #model_candidate[-1].save(train_cfg['model_final_loc'])
                    model_candidate[-1].save(train_cfg['model_loc']+f'_s{pruning_auto_step}.tf')

                    #co = {}
                    #_add_supported_quantized_objects(co)
                    #if pruning_cfg['enable']:
                    #    co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
                    #    model = tf.keras.models.load_model(train_cfg['model_chk_loc'], custom_objects=co)
                    #else:
                    #    model = tf.keras.models.load_model(train_cfg['model_chk_loc'], custom_objects=co)


                else:
                    print("pruning_auto needs train")
                    exit()

                #model_candidate.append(model_pruned)
                #model_good.append(model_pruned)

                evaluate_res = model_candidate[-1].evaluate(train_cfg['X_test'], train_cfg['Y_test'])
                write_line_to_log(log_cfg, f'Keras accuracy = {evaluate_res[1]}')


                #check_sparsity(model, pruning_en)
                allWeightsByLayer = {}

                write_line_to_log(log_cfg, "\n")
                write_line_to_log(log_cfg, "Checking Sparity")
                for layer in model_candidate[-1].layers:
                    if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
                        continue
                    weights=layer.weights[0].numpy().flatten()
                    allWeightsByLayer[layer._name] = weights
                    write_line_to_log(log_cfg, 'Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

                cur_acc.append(float(evaluate_res[1]))

                # calculate the next sparsity rate using dichotomy

                if (pruning_auto_step == 0):
                    cur_p_rate.append(pruning_cfg['start_sparsity'])
                else:
                    acc_delta = cur_acc[-1] - cur_acc[0]

                    if(max(cur_p_rate) == cur_p_rate[-1]):
                        right = 1.0
                        left  = cur_p_rate[-2]
                    elif(max(cur_p_rate) == cur_p_rate[ 1]):
                        right = cur_p_rate[-2]
                        left  = 0.0
                    elif(cur_p_rate[-1] < cur_p_rate[-2]):
                        right = cur_p_rate[-2]
                        left  = cur_p_rate[final_model_num]
                    elif(cur_p_rate[-1] > cur_p_rate[-2]):
                        #right = cur_p_rate[-3]
                        left  = cur_p_rate[-2]

                    if (acc_delta < pruning_cfg['acc_delta_ths']):
                        nxt_sparsity = (cur_p_rate[-1] + left) /2

                        #final_model_num -= 1
                        #model_candidate.pop(-1) # remove the model with large accuracy loss
                        #del model_candidate[-1] # remove the model with large accuracy loss
                        write_line_to_log(log_cfg, f"REMOVE the model in the step : {pruning_auto_step}" )
                        write_line_to_log(log_cfg, f"the pruning rate of the removed model is : {cur_p_rate[-1]}" )

                    else:
                        nxt_sparsity = (cur_p_rate[-1] + right) /2
                        final_model_num = pruning_auto_step
                    cur_p_rate.append(nxt_sparsity)

                # check if it is the end of the search
                pruning_rate_delta = abs(cur_p_rate[-1] - cur_p_rate[-2])
                if (pruning_rate_delta < pruning_cfg['p_rate_delta_ths'] or pruning_cfg['auto'] == 0 or pruning_cfg['enable'] == 0):
                    break # break while()
                # increase the step
                pruning_auto_step += 1
                # increaset the search step
            # log the final accuracy list and pruning list
            write_line_to_log(log_cfg, f"final accuracy list: {str(cur_acc)}" )
            write_line_to_log(log_cfg, f"final pruning rate list: {str(cur_p_rate)}" )
            write_line_to_log(log_cfg, f"optimal model is at the {final_model_num} step" )

            co = {}
            _add_supported_quantized_objects(co)
            if pruning_cfg['enable']:
                co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
                final_model = tf.keras.models.load_model(train_cfg['model_loc']+f'_s{final_model_num}.tf', custom_objects=co)
                final_model  = strip_pruning(final_model)
            else:
                final_model = tf.keras.models.load_model(train_cfg['model_loc']+f'_s{final_model_num}.tf', custom_objects=co)

            return final_model

        return wrapper
    return inner

def pruning_v2(enable, data_len, batch_size, final_sparsity = 0.75):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = func(*args, **kwargs)
            NSTEPS = data_len // batch_size
            if enable:
                def pruneFunction(layer):
                    pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                                                   final_sparsity = final_sparsity,
                                                                                   begin_step = NSTEPS*2,
                                                                                   end_step = NSTEPS*10,
                                                                                   frequency = NSTEPS)
                                     }
                    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.SeparableConv2D):
                        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                    #if isinstance(layer, tf.keras.layers.Dense) and layer.name!='output_dense':
                    if isinstance(layer, tf.keras.layers.Dense):
                        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                    return layer

                model_after = tf.keras.models.clone_model( model, clone_function=pruneFunction)
                model_after = strip_pruning(model_after)
            else:
                model_after = model
            return model_after
        return wrapper
    return inner

def pruning(data_len, batch_size, final_sparsity = 0.5):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = func(*args, **kwargs)
            NSTEPS = data_len // batch_size
            def pruneFunction(layer):
                pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                                               final_sparsity = final_sparsity,
                                                                               begin_step = NSTEPS*2,
                                                                               end_step = NSTEPS*10,
                                                                               frequency = NSTEPS)
                                 }
                if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.SeparableConv2D):
                    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                if isinstance(layer, tf.keras.layers.Dense) and layer.name!='output_dense':

                    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                return layer

            model_pruned = tf.keras.models.clone_model( model, clone_function=pruneFunction)

            return model_pruned
        return wrapper
    return inner


# +
from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm, QConv2D
def quantization(quantizer = "quantized_bits(6,0,alpha=1)" ):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = func(*args, **kwargs)
            print("----- begin quantization -----")
            layers = model.layers
            def _clone_fn(layer):
                if layer.name == layers[-1].name:
                    return layer
                if isinstance(layer, tf.keras.layers.Conv2D):
                    config = layer.get_config()
                    return QConv2D(name = config['name'], filters = config['filters'],
                                                            kernel_size = config["kernel_size"],
                                                            strides=config['strides'],
                                                            padding=config['padding'],
                                                            data_format=config['data_format'],
                                                            dilation_rate=config['dilation_rate'],
                                                            activation=config['activation'],
                                                            use_bias=True,
                                                            kernel_quantizer=quantizer,
                                                            bias_quantizer=quantizer)
                if isinstance(layer, tf.keras.layers.Activation):
                    config = layer.get_config()
                    return QActivation('quantized_relu(6)',name=config['name'])
                if isinstance(layer, tf.keras.layers.Dense):
                    config = layer.get_config()
                    return QDense(units = config['units'],
                                  kernel_initializer='lecun_uniform',
                                  kernel_quantizer=quantizer,
                                  kernel_regularizer=l1(0.0001),
                                  name = config['name'],
                                  use_bias=False)
                return layer
            model_quantized = new_model = clone_model(model, clone_function=_clone_fn)
            model_quantized.summary()
            return model_quantized
        return wrapper
    return inner



# AutoQKeras

from qkeras import print_qstats
import pprint
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import model_quantize

from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import quantized_bits
from qkeras import QDense, QActivation

from qkeras.autoqkeras.utils import print_qmodel_summary
from qkeras.autoqkeras import AutoQKeras

def quantization_auto(prj_cfg, train_cfg, quantization_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            baseline_model = func(*args, **kwargs)
            if not quantization_cfg['enable']:
                return baseline_model

            if not quantization_cfg['auto']:
                return baseline_model # add static quantization

            print("----- begin auto quantization -----")

            # check estimated energy consumption of the QKeras 6-bit model using QTools
            q = run_qtools.QTools(baseline_model,
                                process="horowitz",
                                source_quantizers=[quantized_bits(16, 5, 1)],
                                is_inference=True,
                                weights_path=None,
                                keras_quantizer="fp16",
                                keras_accumulator="fp16",
                                for_reference=False)
            q.qtools_stats_print()

            energy_dict = q.pe(
                weights_on_memory="fixed",
                activations_on_memory="fixed",
                min_sram_size=8*16*1024*1024,
                rd_wr_on_io=False)

            # get stats of energy distribution in each layer
            energy_profile = q.extract_energy_profile(
                qtools_settings.cfg.include_energy, energy_dict)
            # extract sum of energy of each layer according to the rule specified in
            # qtools_settings.cfg.include_energy
            total_energy = q.extract_energy_sum(
                qtools_settings.cfg.include_energy, energy_dict)

            pprint.pprint(energy_profile)
            print()

            print("Total energy: {:.6f} uJ".format(total_energy / 1000000.0))


            quantization_config = {
                    "kernel": {
                            "quantized_bits(2,0,1,alpha=1.0)": 2,
                            "quantized_bits(4,0,1,alpha=1.0)": 4,
                            "quantized_bits(6,0,1,alpha=1.0)": 6,
                            "quantized_bits(8,0,1,alpha=1.0)": 8,
                    },
                    "bias": {
                            "quantized_bits(2,0,1,alpha=1.0)": 2,
                            "quantized_bits(4,0,1,alpha=1.0)": 4,
                            "quantized_bits(6,0,1,alpha=1.0)": 6,
                            "quantized_bits(8,0,1,alpha=1.0)": 8,
                    },
                    "activation": {
                            "quantized_relu(3,1)": 3,
                            "quantized_relu(4,2)": 4,
                            "quantized_relu(8,2)": 8,
                            "quantized_relu(8,4)": 8,
                            "quantized_relu(16,6)": 16
                    },
                    "linear": {
                            "quantized_bits(2,0,1,alpha=1.0)": 2,
                            "quantized_bits(4,0,1,alpha=1.0)": 4,
                            "quantized_bits(6,0,1,alpha=1.0)": 6,
                            "quantized_bits(8,0,1,alpha=1.0)": 8,
                    }
            }

            # These are the layer types we will quantize
            limit = {
                "Dense": [8, 8, 16],
                "Conv2D": [8, 8, 16],
                "Activation": [16],
            }

            # Use this if you want to minimize the model bit size
            goal_bits = {
                "type": "bits",
                    "params": {
                        "delta_p": 8.0, # We tolerate up to a +8% accuracy change
                        "delta_n": 8.0, # We tolerate down to a -8% accuracy change
                        "rate": 2.0,    # We want a x2 times smaller model
                        "stress": 1.0,  # Force the reference model size to be smaller by setting stress<1
                        "input_bits": 8,
                        "output_bits": 8,
                        "ref_bits": 8,
                        "config": {
                            "default": ["parameters", "activations"]
                        }
                    }
            }

            # Use this if you want to minimize the model energy consumption
            goal_energy = {
                "type": "energy",
                "params": {
                    "delta_p": 8.0,
                    "delta_n": 8.0,
                    "rate": 2.0,
                    "stress": 1.0,
                    "process": "horowitz",
                    "parameters_on_memory": ["sram", "sram"],
                    "activations_on_memory": ["sram", "sram"],
                    "rd_wr_on_io": [False, False],
                    "min_sram_size": [0, 0],
                    "source_quantizers": ["fp32"],
                    "reference_internal": "int8",
                    "reference_accumulator": "int32"
                    }
            }

            run_config = {

                    "goal": goal_energy,
                    "quantization_config": quantization_config,
                    "learning_rate_optimizer": False,
                    "transfer_weights": False, # Randomely initialize weights
                    "mode": "bayesian", # This can be bayesian,random,hyperband
                    "seed": 42,
                    "limit": limit,
                    "tune_filters": "layer",
                    "tune_filters_exceptions": "^output",
                    "distribution_strategy": None,
                    # "layer_indexes": range(1, len(baseline_model.layers) - 1),
                    "max_trials": 5 # Let's just do 5 trials for this demonstrator, ideally you should do as many as possible
            }

            if train_cfg['enable']:
                baseline_model.compile(loss=train_cfg['loss'], optimizer=train_cfg['optimizer'], metrics=train_cfg['metrics'])
                autoqk = AutoQKeras(baseline_model, output_dir="autoq/autoq", metrics=["acc"], custom_objects={}, **run_config)
                autoqk.fit(train_cfg['X_train'],
                           train_cfg['Y_train'],
                           # validation_data=(train_cfg['X_test']),
                           validation_split= train_cfg['val_split'],
                           epochs=train_cfg['n_epochs'])

                aqmodel = autoqk.get_best_model()
                print_qmodel_summary(aqmodel)

                # Train for the full epochs
                callbacks = [
                            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
                            ]

                start = time.time()
                history = aqmodel.fit(train_cfg['X_train'],
                                    train_cfg['Y_train'],
                                    # validation_data = train_cfg['Y_train'],
                                    batch_size  = train_cfg['batch_size'],
                                    epochs = train_cfg['n_epochs'],
                                    validation_split= train_cfg['val_split'],
                                    callbacks = callbacks,
                                    verbose=1)
                end = time.time()
                print('\n It took {} minutes to train!\n'.format( (end - start)/60.))

                # This model has some remnants from the optimization procedure attached to it, so let's define a new one
                aqmodel.save_weights("autoqkeras_weights.h5")

                layers = [l for l in aqmodel.layers]
                x = layers[0].output
                for i in range(1, len(layers)):
                    x = layers[i](x)

                new_model = Model(inputs=[layers[0].input], outputs=[x])
                LOSS        = tf.keras.losses.CategoricalCrossentropy()
                OPTIMIZER   = tf.keras.optimizers.Adam(learning_rate=3E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)

                new_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
                new_model.summary()
                new_model.load_weights("autoqkeras_weights.h5")
                print_qmodel_summary(new_model)

            else:
                print("AutoQKeras needs train")
                exit()

            return new_model
        return wrapper
    return inner