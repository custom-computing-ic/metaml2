import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from pathlib import Path
import pprint
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from matplotlib.lines import Line2D

import hls4ml

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import os

from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dense, SeparableConv2D

from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


from sklearn.model_selection import train_test_split
from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from metaml.v1 import *


def getReports(indir):
    data_ = {}

    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))

    if report_vsynth.is_file() and report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))

        # Get the resources from the logic synthesis report
        with report_vsynth.open() as report:
            lines = np.array(report.readlines())
            data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
            data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
            data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
            data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
            data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
            data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
            data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
            data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])

        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_mus']  = float(lat_line.split('|')[2])*5.0/1000.
            data_['latency_ii']   = int(lat_line.split('|')[6])

    return data_

def check_sparsity(model, pruning_en):
    allWeightsByLayer = {}

    write_line_to_log(log_cfg, "\n")
    write_line_to_log(log_cfg, "Checking Sparity")
    write_line_to_log(log_cfg, f"Sparity_en: {pruning_en}")
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue
        weights=layer.weights[0].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        write_line_to_log(log_cfg, 'Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))


parser = argparse.ArgumentParser()
parser.add_argument("-PJN", "--project_name", type=str, default="meta_ml", help="project name")
parser.add_argument("-DN", "--dataset_name", type=str, default="svhn",    help="dataset name")
parser.add_argument("-EP" , "--epochs", type=int,   default=1, help="number of epochs")
parser.add_argument("-T",   "--train",  type=int,   default=1, help="train?")
parser.add_argument(        "--batch_size", type=int,  default=32, help="batch size")
parser.add_argument("-LOG", "--log",    type=int,   default=1, help="log")
parser.add_argument("-SYN", "--synth",  type=int,   default=0, help="Synthesis?")
parser.add_argument("-M",   "--model",  type=str,   default="VGG6", help="model name ")
parser.add_argument("-PRN", "--pruning",type=int,   default=0, help="pruning enable ")
parser.add_argument("-PNR", "--pruning_rate",type=float,default=0.5, help="pruning rate")
parser.add_argument(        "--pruning_auto",type=int,  default=1, help="auto enable ")
parser.add_argument(        "--debug_fast_run",type=int,default=0, help="fast run in debug ")
parser.add_argument("-SCL", "--scale",  type=int,   default=0, help="scale enable ")

args = parser.parse_args()



if __name__ == "__main__":


    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    project_name = args.project_name
    dataset_name = args.dataset_name
    log = args.log
    log_file_name = f"log_{project_name}.txt"
    directory = f"projects/{dataset_name}/{project_name}"

    if dataset_name not in DATASET_POOL:
        print("Dataset is not found")
        print("Select the dataset below")
        print("exit()")
        for key in DATASET_POOL:
            print(key)
        exit()

    pruning_en = args.pruning
    pruning_rate = args.pruning_rate
    scale_en = args.scale

    prj_cfg = {}
    prj_cfg['project_name']  = project_name
    prj_cfg['dataset_name']  = dataset_name
    prj_cfg['directory']     = directory

    prj_cfg['train_en']      = args.train
    prj_cfg['pruning_en']    = pruning_en
    prj_cfg['pruning_rate']  = pruning_rate
    prj_cfg['synthesis_en']  = args.synth

    prj_cfg['log']              = {}
    prj_cfg['log']['enable']    = log
    prj_cfg['log']['log_file']  =  f'{directory}/{log_file_name}'
    log_cfg = prj_cfg['log']

    os.system(f"mkdir -p {directory}")
    start_time = time.strftime("%Y%m%d-%H%M%S")
    write_line_to_log(log_cfg, "\n")
    write_line_to_log(log_cfg, f"Start time: {start_time}")

    write_dic_to_log(log_cfg, prj_cfg)


    if dataset_name == 'jets_hlf':
        X_train, Y_train, X_test, Y_test = load_train_data_jets_hlf()
        input_shape = (16,)
        n_classes = 5
        if(args.model == 'LHC_CNN'):
            X_train = X_train.reshape(X_train.shape[0], 16, 1)
            X_test  = X_test.reshape (X_test.shape[0], 16, 1)
            input_shape = (16,1)

    elif dataset_name == 'cifar10':
        X_train, Y_train, X_test, Y_test = load_train_data_cifar10()
        input_shape, n_classes = get_info_cifar10()
    else:
        X_train, Y_train, X_test, Y_test = load_image_train_data(dataset_name)
        input_shape, n_classes = get_image_info(dataset_name)


    write_line_to_log(log_cfg, f"trainng data size:{str(X_train.shape)}")
    write_line_to_log(log_cfg, f"input_shape: {input_shape}")
    write_line_to_log(log_cfg, f"n_classes:   {n_classes  }")

# pruning strategy
    pruning_cfg = {}
    pruning_cfg['enable']              = pruning_en
    pruning_cfg['auto']                = args.pruning_auto
    pruning_cfg['acc_delta_ths']       = -0.02  # negative, maximum acc loss due to pruning
    pruning_cfg['p_rate_delta_ths']    = 0.02
    if pruning_cfg['auto']:
        pruning_cfg['target_sparsity'] = 1.00
        pruning_cfg['start_sparsity']  = pruning_rate
    else:
        pruning_cfg['target_sparsity'] = pruning_rate
        pruning_cfg['start_sparsity']  = 0.0

    write_line_to_log(log_cfg, "pruning config")
    write_dic_to_log(log_cfg, pruning_cfg)


# scale strategy
    scale_cfg = {}
    scale_cfg['enable']        = scale_en
    scale_cfg['max_trials']    = 5

    write_line_to_log(log_cfg, "scale config")
    write_dic_to_log(log_cfg, scale_cfg)


    train_cfg = {}
    train_cfg['enable']     = args.train
    train_cfg['n_epochs']   = args.epochs
    train_cfg['batch_size'] = args.batch_size

    train_cfg['patience']   = 50
    train_cfg['pruning_en'] = pruning_en
    train_cfg['scale_en']   = scale_en
    train_cfg['val_split']  = 0.1
    train_cfg['model_chk_loc']  = f'{directory}/{args.model}_chk_baseline_model.h5'
    train_cfg['model_final_loc']  = f'{directory}/{args.model}_final_baseline_model.h5'
    train_cfg['model_weights_loc']  = f'{directory}/{args.model}_final_weights_model.h5'
    train_cfg['model_loc']  = f'{directory}/{args.model}_model'

    write_line_to_log(log_cfg, "part of training config")
    write_dic_to_log(log_cfg, train_cfg)


    #n_epochs = args.epochs
    #patience = 50

    # early stopping callback
    es = EarlyStopping(monitor='val_loss', patience=train_cfg['patience'])
    # Learning rate scheduler
    ls = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=int(train_cfg['patience']/3))

    # model checkpoint callback
    # this saves our model architecture + parameters into mlp_model.h5
    chkp = ModelCheckpoint(train_cfg['model_chk_loc'],
                monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=False, mode='auto',
                save_freq='epoch')
    prn = pruning_callbacks.UpdatePruningStep()


    if pruning_en:
        #train_cfg['callbacks']  = [ es, ls, chkp, prn ]
        train_cfg['callbacks']  = [ ls, prn ]
    else:
        train_cfg['callbacks']  = [ es, ls]
    #else:
    #    train_cfg['callbacks']  = [ es, ls, chkp ]

    if args.debug_fast_run == 1:
        train_cfg['X_train']    = X_train[:40960]
        train_cfg['Y_train']    = Y_train[:40960]
    else:
        train_cfg['X_train']    = X_train
        train_cfg['Y_train']    = Y_train

    train_cfg['X_test']     = X_test
    train_cfg['Y_test']     = Y_test
    train_cfg['loss']       = tf.keras.losses.CategoricalCrossentropy()

    train_cfg['optimizer']  = tf.keras.optimizers.Adam(learning_rate=3E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    #train_cfg['optimizer']  = tf.keras.optimizers.Adam(learning_rate=3E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    train_cfg['metrics']    = ["accuracy"]



    @strategy.pruning_pretrain_auto(pruning_cfg, prj_cfg, train_cfg)
    @strategy.scale_auto_v3(scale_cfg, prj_cfg, train_cfg)
    def prepare_model():

        if args.model in MODEL_POOL:
            model_config = MODEL_POOL[args.model]
        else:
            write_line_to_log(log_cfg, "Model is not found")
            write_line_to_log(log_cfg, "Select the model below")
            write_line_to_log(log_cfg, "exit()")
            for key in MODEL_POOL:
                print(key)
            exit()


        write_line_to_log(log_cfg, "\n")
        write_line_to_log(log_cfg, f"Model name: {args.model}")
        write_line_to_log(log_cfg, f"Model name, cnn filters, fc num: {str(model_config)}")

        model = model_config[0](input_shape, n_classes, model_config[1])
        model.summary()

        for layer in model.layers:
            if layer.__class__.__name__ in ['Conv2D', 'Dense']:
                w = layer.get_weights()[0]
                layersize = np.prod(w.shape)
                write_line_to_log(log_cfg, "{}: w shape is {}".format(layer.name,str(w.shape))) # 0 = weights, 1 = biases
                write_line_to_log(log_cfg, "{}: w size is {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
                if (layersize > 4096 and args.synth): # assuming that shape[0] is batch, i.e., 'None'
                #if (layersize > 8192): # assuming that shape[0] is batch, i.e., 'None'
                    write_line_to_log(log_cfg, "Layer {} is too large ({}) for vivado HLS".format(layer.name,layersize))
                    write_line_to_log(log_cfg, "exit()")
                    exit()

        return model


    model = prepare_model()
    print("model after strategy: ")
    model.summary()

# train

#    train = args.train # True if you want to retrain, false if you want to load a previsously trained model
#    n_epochs = args.epochs
#    patience = 50
#
#    if train:
#
#        if args.model != 'ResNet18':
#            LOSS        = tf.keras.losses.CategoricalCrossentropy()
#            OPTIMIZER   = tf.keras.optimizers.Adam(learning_rate=3E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
#            model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
#
#
#
#        history = model.fit(X_train,
#                  Y_train,
#                  #batch_size=args.batch,
#                  epochs = n_epochs,
#                  validation_split=0.1,
#                  callbacks = callbacks)
#
#        #print(history.history.keys())
#        write_line_to_log(log_cfg, "\n\n")
#        write_line_to_log(log_cfg, "training history:")
#        write_dic_to_log(log_cfg, history.history)
#        #model.save(f'{directory}/baseline_model.h5')
#
#    else: # no train
#        if pruning_en:
#            co = {}
#            _add_supported_quantized_objects(co)
#            co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
#            #model = tf.keras.models.load_model('quantized_pruned_cnn_model.h5')
#            model = tf.keras.models.load_model(f'{directory}/{args.model}_baseline_model.h5', custom_objects=co)
#        else:
#            model = tf.keras.models.load_model(f'{directory}/{args.model}_baseline_model.h5')

    check_sparsity(model, pruning_en)

    #evaluate_res = model.evaluate(X_test, Y_test)
    y_keras    = model.predict(X_test)
    accuracy_keras = float(accuracy_score (np.argmax(Y_test, axis=1), np.argmax(y_keras,axis=1)))

    write_line_to_log(log_cfg, f'Full test dataset accuracy')
    write_line_to_log(log_cfg, f'Keras  accuracy = {accuracy_keras}')

    model.save(train_cfg['model_final_loc'])


#hls4ml
    #if pruning_en:
    #    model  = strip_pruning(model)

    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    #First, the baseline model
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='ap_fixed<16,6>')

    # Set the precision and reuse factor for the full model
    #hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config['Model']['ReuseFactor'] = 1
    hls_config['Model']['Strategy'] = 'Latency'
    hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'

    plotting.print_dict(hls_config)

    #exit()

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = model
    cfg['OutputDir']  = f'{directory}/{project_name}'
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'


    write_line_to_log(log_cfg, "\n\n")
    write_line_to_log(log_cfg, "hls4ml config:")
    write_dic_to_log(log_cfg, cfg)

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    # -
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=f'{directory}/hls4ml_plot_{project_name}.png')
    tf.keras.utils.plot_model(model, to_file=f'{directory}/keras_plot_{project_name}.png')


# test accuracy
    test_num = 3000
    X_test = X_test[:test_num]
    Y_test = Y_test[:test_num]
    y_keras = model.predict(X_test)
    y_hls = hls_model.predict(np.ascontiguousarray(X_test))

    accuracy_keras = float(accuracy_score (np.argmax(Y_test, axis=1), np.argmax(y_keras,axis=1)))
    accuracy_hls   = float(accuracy_score (np.argmax(Y_test, axis=1), np.argmax(y_hls,axis=1)))

    write_line_to_log(log_cfg, f'Test accuracy with {test_num} samples')
    write_line_to_log(log_cfg, f'Keras  accuracy = {accuracy_keras}')
    write_line_to_log(log_cfg, f'HLS4ML accuracy = {accuracy_hls}')


    synth = args.synth # Only if you want to synthesize the models yourself (>1h per model) rather than look at the provided reports.
    if synth:
        hls_model.build(csim=False, synth=True, vsynth=True)

        data_pruned_ref = getReports(f'{directory}/{project_name}')

        print(f"\n Resource usage and latency: {project_name}")
        pprint.pprint(data_pruned_ref)
        write_dic_to_log(log_cfg, data_pruned_ref)
    # -


