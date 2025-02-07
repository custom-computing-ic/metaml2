from functools import wraps

from tensorflow.keras.models import Model, clone_model
import tensorflow.compat.v2 as tf

from bayes_opt import BayesianOptimization as BOptimize
import sys

from .utility import *

from tensorflow.keras.regularizers import l1
def scaling_model(model, scaling_rate, train_cfg):
    scaled_model = tf.keras.Sequential()

    # print("begin with scaling rate: ")
    # print(str(scaling_rate))

    # for layer in model.layers:
    #     # print(layer.get_config())
    #     # Check if the layer is a Conv2D layer
    #     if isinstance(layer, tf.keras.layers.Conv2D):
    #         print(f'{layer.name} is Conv2D')
    #         # Scale down the filters by a factor of 2
    #         new_filters = max(1, int(layer.filters * scaling_rate))

    #         # Create a new Conv2D layer with scaled-down filters
    #         new_layer = tf.keras.layers.Conv2D(name = layer.name,
    #                                         filters = new_filters,
    #                                         kernel_size = layer.kernel_size,
    #                                         strides = layer.strides,
    #                                         activation = layer.activation,
    #                                         padding=layer.padding,
    #                                         kernel_initializer=layer.kernel_initializer,
    #                                         kernel_regularizer = layer.kernel_regularizer,
    #                                         use_bias = layer.use_bias
    #                                         )
    #         # Add the new layer to the new model
    #         scaled_model.add(new_layer)

    #     elif isinstance(layer, tf.keras.layers.Dense):
    #         print(f'{layer.name} is Dense')
    #         config = layer.get_config()
    #         new_unit = max(1, int(config['units'] * scaling_rate)) if (layer.name!='output_dense') else config['units']

    #         # print(f"layer {str(config['name'])} scale from {str(config['units'])} to {str(new_unit)}")
    #         new_layer = tf.keras.layers.Dense(name = config['name'], 
    #                                     units = new_unit, 
    #                                     activation = config['activation'],  
    #                                     kernel_initializer=layer.kernel_initializer,
    #                                     kernel_regularizer = layer.kernel_regularizer,
    #                                     use_bias=config['use_bias'])
    #         scaled_model.add(new_layer)

    #     elif isinstance(layer, tf.keras.layers.BatchNormalization):
    #         print(f'{layer.name} is BN')
    #         scaled_model.add(tf.keras.layers.BatchNormalization())

    #     elif isinstance(layer, tf.keras.layers.Activation):
    #         print(f'{layer.name} is Activation')
    #         scaled_model.add(tf.keras.layers.Activation(activation = layer.activation,
    #                                                     name = layer.name))
            
    #     elif isinstance(layer, tf.keras.layers.MaxPool2D):
    #         print(f'{layer.name} is MaxPool2D')
    #         scaled_model.add(tf.keras.layers.MaxPool2D(name = layer.name,
    #                                                     pool_size = layer.pool_size))
            
    #     elif isinstance(layer, tf.keras.layers.Flatten):
    #         print(f'{layer.name} is Flatten')
    #         scaled_model.add(tf.keras.layers.Flatten())
            
    #     else:
    #         # For non-Conv2D layers, add them to the new model as is
    #         print(f'{layer.name} is else')
    #         scaled_model.add(layer)
    #         scaled_model.summary()


    def _clone_fn(layer):
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.Conv2D):
            #print(config['name'])
            #print(origin_layers_dic[config['name']])
            new_filters = max(1, int(layer.filters * scaling_rate))
            return tf.keras.layers.Conv2D(
                            name = config['name'],
                            filters = new_filters,
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
            new_unit = max(1, int(config['units'] * scaling_rate)) if (layer.name!='output_dense') else config['units']
            return tf.keras.layers.Dense(
                            name = config['name'],
                            units = new_unit,
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

    new_model = tf.keras.models.clone_model(model, clone_function=_clone_fn)                
    new_model.summary()
    return new_model

def scaling_score(model, param=1): 
    return -model.count_params()

def pruning_model(model, pruning_rate, train_cfg):
    # model.summary()
        
    print(f"begin with pruning rate: {str(pruning_rate)}")
    
    def pruneFunction(layer):
        NSTEPS = len(train_cfg['X_train']) // train_cfg['batch_size']

        pruning_params = {'pruning_schedule': 
            sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                    final_sparsity = float(pruning_rate),
                                    begin_step = NSTEPS*2, 
                                    end_step = NSTEPS*6, 
                                    frequency = NSTEPS)
                            }
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        if isinstance(layer, tf.keras.layers.Conv1D): 
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer
    
    model_pruned = tf.keras.models.clone_model(model, clone_function=pruneFunction)
    # # model_pruned.summary()
    # model_pruned = strip_pruning(model_pruned)
    return model_pruned

# @strategy.pruning_pretrain_auto(pruning_cfg, prj_cfg, train_cfg)
# def pruning_model_by_acc_th(model, max_accuracy_loss, train_cfg):
#     # model.summary()
        
#     print(f"begin with max_accuracy_loss: {str(max_accuracy_loss)}")
#     return model

def pruning_score(model, param): 
    return param # pruning_rate

def sole_strategy_BayesianOptimization(optimization_func, score_func, prj_cfg, train_cfg, param_min=0.5, param_max=0.99):
    # sole_strategy_BayesianOptimization(scaling_model, scaling_score, prj_cfg, train_cfg)

    # optimization_func(model, scaling_rate)
    # score_func(model) -> model.count_params()

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            # get original accuracy score
            model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            model.fit(
                    train_cfg['X_train'], 
                    train_cfg['Y_train'],
                    batch_size  = train_cfg['batch_size'],
                    epochs      = train_cfg['n_epochs'],
                    validation_split= train_cfg['val_split'],
                    callbacks   = train_cfg['callbacks'])   
            orgn_accuracy_score = model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
            orgn_accuracy_score = orgn_accuracy_score[1]

            min_accuracy = orgn_accuracy_score * (1 - train_cfg['acc_threshold'])
            write_line_to_log(log_cfg, f"minimum accuracy is {str(min_accuracy)}")

            def black_box_function(param):
                write_line_to_log(log_cfg, "\n")
                write_line_to_log(log_cfg, f"iteration with parameter : param={str(param)}")
                final_model = optimization_func(model, param, train_cfg=train_cfg)

                final_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                final_model.fit(
                                train_cfg['X_train'], 
                                train_cfg['Y_train'],
                                batch_size  = train_cfg['batch_size'],
                                epochs      = train_cfg['n_epochs'],
                                validation_split= train_cfg['val_split'],
                                callbacks   = train_cfg['callbacks'])   
                
                # test model
                accuracy_score = final_model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
                accuracy_score = accuracy_score[1]
                
                # check model layer size
                final_model.summary()
                
                # check_sparsity
                allWeightsByLayer = {}
                
                write_line_to_log(log_cfg, "\n")
                write_line_to_log(log_cfg, "Checking Sparity")
                for layer in final_model.layers:
                    if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
                        continue
                    weights=layer.weights[0].numpy().flatten()
                    allWeightsByLayer[layer._name] = weights
                    write_line_to_log(log_cfg, 'Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

                if accuracy_score < min_accuracy:
                    return -sys.maxsize
                
                model_score = score_func(final_model, param)  
                write_line_to_log(log_cfg, f"accuracy is {str(accuracy_score)}")
                write_line_to_log(log_cfg, f"score = {str(model_score)}")
                return model_score

            params_nn ={
                'param': (param_min, param_max)
            }
            # nn_bo = BayesianOptimization(black_box_function, params_nn, random_state=111)
            # nn_bo.maximize(init_points=25, n_iter=4)
            
            optimizer = BOptimize(
                f=black_box_function,
                pbounds=params_nn,
                random_state=1,
                allow_duplicate_points=True
            )

            optimizer.maximize(init_points=2, n_iter=train_cfg['bayesian_iter'])
            params_nn_ = optimizer.max['params']

            write_line_to_log(log_cfg, f"Bayesian Optimization final param={str(params_nn_['param'])}")
            
            final_model = optimization_func(model, params_nn_['param'], train_cfg)
            # final_model = strip_pruning(final_model)
            return final_model

        return wrapper
    return inner

import numpy as np
def check_sparsity(model):
    allWeightsByLayer = {}
    
    print( "\n")
    print( "Checking Sparity")
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue
        weights=layer.weights[0].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print( 'Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))


from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow_model_optimization as tfmot
def scale_prune_BayesianOptimization(scale_cfg, pruning_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            # get original accuracy score
            model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            model.fit(
                    train_cfg['X_train'], 
                    train_cfg['Y_train'],
                    batch_size  = train_cfg['batch_size'],
                    epochs      = train_cfg['n_epochs'],
                    validation_split= train_cfg['val_split'],
                    callbacks   = train_cfg['callbacks'])   
            orgn_accuracy_score = model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
            orgn_accuracy_score = orgn_accuracy_score[1]
            orgn_model_size = model.count_params()

            min_accuracy = orgn_accuracy_score * (1 - scale_cfg['threshold'])
            write_line_to_log(log_cfg, f"minimum accuracy is {str(min_accuracy)}")

            
            def black_box_function(pruning_rate, scaling_rate):
                write_line_to_log(log_cfg, "\n")
                write_line_to_log(log_cfg, f"iteration with parameter : scaling_rate={str(scaling_rate)}, pruning_rate={str(pruning_rate)}")
                # write_line_to_log(log_cfg, f"iteration with parameter : pruning_rate={str(pruning_rate)}")

                # scale model
                scaled_model = scaling_model(model, scaling_rate, train_cfg)
                write_line_to_log(log_cfg, f"scaled model parameter: {str(scaled_model.count_params())}")

                # scaled_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                # scaled_model.fit(
                #                 train_cfg['X_train'], 
                #                 train_cfg['Y_train'],
                #                 batch_size  = train_cfg['batch_size'],
                #                 epochs      = train_cfg['n_epochs'],
                #                 validation_split= train_cfg['val_split'],
                #                 callbacks   = train_cfg['callbacks'])   

                # prune model
                final_model = pruning_model(scaled_model, pruning_rate, train_cfg)

                final_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                final_model.fit(
                                train_cfg['X_train'], 
                                train_cfg['Y_train'],
                                batch_size  = train_cfg['batch_size'],
                                epochs      = train_cfg['n_epochs'],
                                validation_split= train_cfg['val_split'],
                                callbacks   = train_cfg['callbacks'])   
                
                final_model = strip_pruning(final_model)
                final_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                

                # test model
                accuracy_score = final_model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
                accuracy_score = accuracy_score[1]

                if accuracy_score < min_accuracy:
                    return -sys.maxsize
                
                model_size = scaled_model.count_params()  # TODO: how to estimate model utilization? 
                scale_score = (orgn_model_size - model_size) / orgn_model_size # (0, 1) # maximize
                prune_score = pruning_rate # (0, 1) maximize
                
                write_line_to_log(log_cfg, f"scale_score = {str(scale_score)}" )
                write_line_to_log(log_cfg, f"prune_score = {str(prune_score)}")
                
                return (scale_score + prune_score) # TODO: how to design the scorer? and the weights?     
                # return prune_score

            params_nn ={
                'scaling_rate': (0.125, 0.99), # should be (0, 1]
                'pruning_rate': (0.5, 0.99) # should be [0, 1)
            }
            
            optimizer = BOptimize(f=black_box_function, pbounds=params_nn, random_state=1)

            optimizer.maximize(init_points=2, n_iter=train_cfg['bayesian_iter'])
            write_line_to_log(log_cfg, "Bayesian Optimization final parameters: ")

            params_nn_ = optimizer.max['params']
            write_line_to_log(log_cfg, f"scaling_rate={str(params_nn_['scaling_rate'])}, pruning_rate={str(params_nn_['pruning_rate'])}")

            scaled_model = scaling_model(model, params_nn_['scaling_rate'], train_cfg)
            final_model = pruning_model(scaled_model, params_nn_['pruning_rate'], train_cfg)

            # final_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            # final_model.fit(
            #                 train_cfg['X_train'], 
            #                 train_cfg['Y_train'],
            #                 batch_size  = train_cfg['batch_size'],
            #                 epochs      = train_cfg['n_epochs'],
            #                 validation_split= train_cfg['val_split'],
            #                 callbacks   = train_cfg['callbacks'])   
            
            # final_model = strip_pruning(final_model)
            # final_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            return final_model
        return wrapper
    return inner


def prune_by_iteration(model, param, train_cfg, prj_cfg):
    # the loop will not break if accuracy loss exeeds threshold
    # will break if the pruning rate delta < 'pruning_rate_delta'
    log_cfg = prj_cfg['log']
        
    NSTEPS = len(train_cfg['X_train']) // train_cfg['batch_size']
    
    write_line_to_log(log_cfg, "start prune_by_iteration")
    pruning_auto_step = 0
    final_model_num = 0
    cur_acc     = []
    cur_p_rate  = [0.0]
        
    acc_delta   = 1.0
    pruning_rate_delta = 1.0
    model_candidate = []
    model_good = []

    acc_delta_ths = -0.02
    p_rate_delta_ths = 0.02

    # to check if iter_param is already trained --> 没必要，因为前面的都已经训了，如果以iteration作为param，会有重复训练

    while pruning_auto_step < param:
        write_line_to_log(log_cfg, f"pruning step: {pruning_auto_step}")
        write_line_to_log(log_cfg, f"current accuracy list: {str(cur_acc)}" )
        write_line_to_log(log_cfg, f"current pruning rate list: {str(cur_p_rate)}" )
        write_line_to_log(log_cfg, "\n" )
        write_line_to_log(log_cfg, f"last acc delta: {acc_delta}")
        write_line_to_log(log_cfg, f"last pruning rate delta: {pruning_rate_delta}")
        write_line_to_log(log_cfg, f"acc delta threshold: {acc_delta_ths}")
        write_line_to_log(log_cfg, f"pruning rate delta threshold: {p_rate_delta_ths}")

        write_line_to_log(log_cfg, "\n" )
        write_line_to_log(log_cfg, f"curent pruning rate: {cur_p_rate[-1]}")
            
        
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

            #model_candidate[-1].save(train_cfg['model_final_loc'])
            model_candidate[-1].save(train_cfg['model_loc']+f'_pruned_s{pruning_auto_step}.h5')

        else:
            print("pruning_auto needs train")
            exit()

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
                left  = cur_p_rate[-3]
            elif(cur_p_rate[-1] > cur_p_rate[-2]):
                right = cur_p_rate[-3]
                left  = cur_p_rate[-2]
            
            # if (acc_delta < pruning_cfg['acc_delta_ths']):
            #     nxt_sparsity = (cur_p_rate[-1] + left) /2

            #     #final_model_num -= 1
            #     #model_candidate.pop(-1) # remove the model with large accuracy loss
            #     #del model_candidate[-1] # remove the model with large accuracy loss
            #     write_line_to_log(log_cfg, f"REMOVE the model in the step : {pruning_auto_step}" )
            #     write_line_to_log(log_cfg, f"the pruning rate of the removed model is : {cur_p_rate[-1]}" )
            
            nxt_sparsity = (cur_p_rate[-1] + right) /2
            final_model_num = pruning_auto_step 
            cur_p_rate.append(nxt_sparsity)

        # check if it is the end of the search 
        pruning_rate_delta = abs(cur_p_rate[-1] - cur_p_rate[-2])
        if (pruning_rate_delta < p_rate_delta_ths):
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
        final_model = tf.keras.models.load_model(train_cfg['model_loc']+f'_pruned_s{final_model_num}.h5', custom_objects=co)
        final_model  = strip_pruning(final_model)
    else:
        final_model = tf.keras.models.load_model(train_cfg['model_loc']+f'_pruned_s{final_model_num}.h5', custom_objects=co)
    
    return final_model, cur_p_rate[final_model_num]


def black_box_process(model, param, train_cfg):
    (model, score) = prune_by_iteration(model, param, train_cfg)
    return (model, score)

def BayesianOptimization_template(scale_cfg, pruning_cfg, prj_cfg, train_cfg):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_cfg = prj_cfg['log']
            model = func(*args, **kwargs)

            # get original accuracy score
            model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            model.fit(
                    train_cfg['X_train'], 
                    train_cfg['Y_train'],
                    batch_size  = train_cfg['batch_size'],
                    epochs      = train_cfg['n_epochs'],
                    validation_split= train_cfg['val_split'],
                    callbacks   = train_cfg['callbacks'])   
            orgn_accuracy_score = model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
            orgn_accuracy_score = orgn_accuracy_score[1]

            min_accuracy = orgn_accuracy_score * (1 - scale_cfg['threshold'])
            write_line_to_log(log_cfg, f"original accuracy is {str(orgn_accuracy_score)}, minimum accuracy is {str(min_accuracy)}")

            def pruning_model(model, pruning_rate):
                # model.summary()
                    
                # print(f"begin with pruning rate: {str(pruning_rate)}")
                
                def pruneFunction(layer):
                    NSTEPS = len(train_cfg['X_train']) // train_cfg['batch_size']

                    pruning_params = {'pruning_schedule': 
                        sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                final_sparsity = float(pruning_rate),
                                                begin_step = NSTEPS*2, 
                                                end_step = NSTEPS*6, 
                                                frequency = NSTEPS)
                                        }
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                    if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
                        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                    return layer
                
                model_pruned = tf.keras.models.clone_model(model, clone_function=pruneFunction)
                # model_pruned.summary()
                model_pruned = strip_pruning(model_pruned)
                return model_pruned

            # def black_box_function(scaling_rate, pruning_rate):
            def black_box_function(param1):
                write_line_to_log(log_cfg, "\n")
                # write_line_to_log(log_cfg, f"iteration with parameter : scaling_rate={str(scaling_rate)}, pruning_rate={str(pruning_rate)}")
                write_line_to_log(log_cfg, f"iteration with parameter : pruning_rate={str(pruning_rate)}")

                # process model 
                (final_model, score) = black_box_process(model, param1, train_cfg)

                # test model
                accuracy_score = final_model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
                accuracy_score = accuracy_score[1]

                if accuracy_score < min_accuracy:
                    return -sys.maxsize
                
                # define score
                # model_size = scaled_model.count_params()  # TODO: how to estimate model utilization? 
                # scale_score = (orgn_model_size - model_size) / orgn_model_size # (0, 1) # maximize
                
                # write_line_to_log(log_cfg, f"scale_score = {str(scale_score)}" )
                write_line_to_log(log_cfg, f"score = {str(score)}")
                
                # return (scale_score * 5 + prune_score) # TODO: how to design the scorer? and the weights?     
                return score

            params_nn ={
                'param1': (1, 10)
            }
            
            optimizer = BOptimize(f=black_box_function, pbounds=params_nn, random_state=1)

            optimizer.maximize(init_points=2, n_iter=3)
            write_line_to_log(log_cfg, "Bayesian Optimization final parameters: ")

            params_nn_ = optimizer.max['params']
            # write_line_to_log(log_cfg, f"scaling_rate={str(params_nn_['scaling_rate'])}, pruning_rate={str(params_nn_['pruning_rate'])}")

            # progress model
            final_model = black_box_process(model, params_nn_['param1'], train_cfg)

            # final_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            # final_model.fit(
            #                 train_cfg['X_train'], 
            #                 train_cfg['Y_train'],
            #                 batch_size  = train_cfg['batch_size'],
            #                 epochs      = train_cfg['n_epochs'],
            #                 validation_split= train_cfg['val_split'],
            #                 callbacks   = train_cfg['callbacks'])   
            
            # final_model = strip_pruning(final_model)
            # final_model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            return final_model
        return wrapper
    return inner



class OptBlock:
    def __init__(self):
        pass

    def process_model(self, model, param, train_cfg): # return (model, score)
        pass

    def get_score(model, weight=1, param=0.5):
        pass

    def train_model(self, model, train_cfg): 
        model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(
                train_cfg['X_train'], 
                train_cfg['Y_train'],
                batch_size  = train_cfg['batch_size'],
                epochs      = train_cfg['n_epochs'],
                validation_split= train_cfg['val_split'],
                callbacks   = train_cfg['callbacks'])   
        return model
    
    def evaluate_model_accuracy(self, model, train_cfg):
        accuracy = model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
        accuracy = accuracy[1]
        return accuracy
        

class ScaleByFactor(OptBlock):
    def __init__(self, param):
        self.name = "OPT_scale_by_factor"
        self.param = param

class PruneByRate(OptBlock):
    def __init__(self, param):
        self.name = "OPT_prune_by_rate"
        self.param = param

class QuantizationByQuantizer(OptBlock):
    def __init__(self, param):
        self.name = "OPT_quantization_by_quantizer"
        self.param = param


def parse_pipeline(model, pipeline, train_cfg):
    # pipeline = {
    #       "instruction":  ["S", "P", "Q"],
    #       "parameters": [0.5, 0.6, (2, 0, 1) ]
    # 

    opt_dict = {
        "S": ScaleByFactor,
        "P": PruneByRate,
        "Q": QuantizationByQuantizer,
    }

    instruction = pipeline["instruction"]
    params = pipeline["params"]
    opt_pipeline = []

    for index in range(len(instruction)):
        if instruction[index] in opt_dict:
            # get O-block
            opt = opt_dict[instruction[index]](params[index])
            opt_pipeline.append(opt)

    def black_box_function(param1, param2, param3): # //// ------------------------------------------------- 需要是params
        # write_line_to_log(log_cfg, "\n")
        # write_line_to_log(log_cfg, f"iteration with parameter : scaling_rate={str(scaling_rate)}, pruning_rate={str(pruning_rate)}")

        model = OptBlock.train_model(model=model, train_cfg=train_cfg)
        orgn_accuracy_score = OptBlock.evaluate_model_accuracy(model=model, train_cfg=train_cfg)

        min_accuracy = orgn_accuracy_score * (1 - train_cfg['threshold'])
        print(f"minimum accuracy is {str(min_accuracy)}")

        params_list = [param1, param2, param3]
        score = 0
        for i in range(len(opt_pipeline)):
            opt = opt_pipeline[i]
            model = opt.process_model(model, params_list[i], train_cfg)
            score += opt.get_score(model, weight=1)
            
        model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
        # test model
        accuracy_score = model.evaluate(train_cfg['X_test'], train_cfg['Y_test'])
        accuracy_score = accuracy_score[1]

        if accuracy_score < min_accuracy:
            return -sys.maxsize
        
        print(f"score = {str(score)}" )

        return score # TODO: how to design the scorer? and the weights?     

    params_nn = {
        "param1": params[0],
        "param2": params[1],
        "param3": params[2]
    }

    optimizer = BOptimize(f=black_box_function, pbounds=params_nn, random_state=1)

    optimizer.maximize(init_points=2, n_iter=3)
    # write_line_to_log(log_cfg, "Bayesian Optimization final parameters: ")

    params_nn_ = optimizer.max['params']
    # write_line_to_log(log_cfg, f"scaling_rate={str(params_nn_['scaling_rate'])}, pruning_rate={str(params_nn_['pruning_rate'])}")

    # progress model
    for opt in opt_pipeline:
        model, _ = opt.process_model(model, train_cfg)
    
    model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    

