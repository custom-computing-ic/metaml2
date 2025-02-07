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
parser.add_argument("-T",   "--train",  type=int,   default=0, help="train?")
parser.add_argument(        "--batch_size", type=int,  default=32, help="batch size")
parser.add_argument("-LOG", "--log",    type=int,   default=1, help="log")
parser.add_argument("-SYN", "--synth",  type=int,   default=1, help="Synthesis?")
parser.add_argument("-M",   "--model",  type=str,   default="VGG6", help="model name ")
parser.add_argument("-PRN", "--pruning",type=int,   default=0, help="pruning enable ")
parser.add_argument("-PNR", "--pruning_rate",type=float,default=0.5, help="pruning rate")
parser.add_argument(        "--pruning_auto",type=int,  default=1, help="auto enable ")
parser.add_argument(        "--debug_fast_run",type=int,default=0, help="fast run in debug ")
parser.add_argument("-SCL", "--scale",  type=int,   default=0, help="scale enable ")

parser.add_argument("-LD",  "--load_model",  type=str , help="model location ")
parser.add_argument("-IO",  "--io",  type=str , default='s',  help="io_stream ")
parser.add_argument("-FPGA",  "--fpga_name",  type=str , default='U250',  help="FPGA name ")

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

    #np.save("X_test.npy", X_test)
    #np.save("Y_test.npy", Y_test)
    #exit(1)

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



    co = {}
    _add_supported_quantized_objects(co)
    if pruning_cfg['enable']:
        co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
        model = tf.keras.models.load_model(args.load_model, custom_objects=co)
        model  = strip_pruning(model)
    else:
        model = tf.keras.models.load_model(args.load_model, custom_objects=co)

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



#hls4ml
    #if pruning_en:
    #    model  = strip_pruning(model)

    if(args.model=='ResNet8' or args.model == 'LHC_CNN'):
        hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
        hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
        hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    #First, the baseline model
    #default_precision = 'ap_fixed<32,16>'
    default_precision = 'ap_fixed<18,8>'
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision=default_precision)

    # Set the precision and reuse factor for the full model
    #hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config['Model']['ReuseFactor'] = 1
    hls_config['Model']['Strategy'] = 'Latency'
    hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'

    plotting.print_dict(hls_config)

    #exit()

    cfg = hls4ml.converters.create_config(backend='Vivado')
    if args.io == 'p':
        cfg['IOType']     = 'io_parallel' # Must set this if using CNNs!
    else:
        cfg['IOType']     = 'io_stream' # Must set this if using CNNs!


    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = model
    cfg['OutputDir']  = f'{directory}/{project_name}'

    fpga_name   = args.fpga_name
    fpga_part   = get_fpga_part(fpga_name)

    #cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'
    cfg['XilinxPart'] = fpga_part

    if(fpga_name=='Z7020' or fpga_name=='A200' or fpga_name=='Z7045'):
        cfg['ClockPeriod'] = 10
    else:
        cfg['ClockPeriod'] = 5



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
        #hls_model.build(csim=False, synth=True, vsynth=True)
        hls_model.build(csim=False, synth=True)

        data_pruned_ref = getReports(f'{directory}/{project_name}')

        print(f"\n Resource usage and latency: {project_name}")
        pprint.pprint(data_pruned_ref)
        write_dic_to_log(log_cfg, data_pruned_ref)
    # -


