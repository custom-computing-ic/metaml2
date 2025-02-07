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
# import plotting

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import os

# MY
import os
os.environ['PATH'] = '/mnt/ccnas2/bdp/opt/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# import matplotlib
# import matplotlib.pyplot as plt
import pdb
# pdb.set_trace()


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
parser.add_argument("-SCT", "--scaling_trial",  type=int,   default=5, help="scale max_trials")
parser.add_argument("-SCA", "--scaling_auto",  type=int,   default=1, help="auto scaling enable ")
parser.add_argument("-SCR", "--scaling_step",  type=int,   default=8, help="auto scaling step ")
parser.add_argument(        "--scaling_rate",  type=float,   default=0.5, help="static scaling rate ")

parser.add_argument(        "--accuracy_threshold",  type=float,   default=0.02, help="optimization accuracy threshold ")
parser.add_argument(        "--bayesian_iter",  type=int,   default=10, help="Bayesian optimization iteration")

parser.add_argument("-Q", "--quantization_enable",  type=int,   default=0, help="quantization enable ")
parser.add_argument("-QA", "--quantization_auto",  type=int,   default=1, help="auto enable ")

parser.add_argument("-RF", "--reuse_factor",  type=int,   default=1, help="hls4ml reuse factor ")
parser.add_argument(       "--prev_prj_name",  type=str,   default="", help="previous project name ")

args = parser.parse_args()




if __name__ == "__main__":

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    #     except RuntimeError as e:
    #         print(e)

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
    prj_cfg['model']  = args.model

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


# scale strategy
    scale_cfg = {}
    scale_cfg['enable']        = scale_en
    scale_cfg['max_trials']    = args.scaling_trial
    scale_cfg['auto']          = args.scaling_auto
    scale_cfg['step']          = args.scaling_step # rate = 1 / step
    scale_cfg['rate']          = args.scaling_rate # rate = 1 / step
    scale_cfg['patience']      = 3
    scale_cfg['threshold']     = args.accuracy_threshold

    write_line_to_log(log_cfg, "scale config")
    write_dic_to_log(log_cfg, scale_cfg)

    # quantization strategy
    quantization_cfg = {}
    quantization_cfg['enable'] = args.quantization_enable
    quantization_cfg['auto'] = args.quantization_auto

    write_line_to_log(log_cfg, "quantization config")
    write_dic_to_log(log_cfg, quantization_cfg)

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

    train_cfg['acc_threshold'] = args.accuracy_threshold
    train_cfg['bayesian_iter'] = args.bayesian_iter

    write_line_to_log(log_cfg, "part of training config")
    write_dic_to_log(log_cfg, train_cfg)

    #n_epochs = args.epochs
    #patience = 50

    # early stopping callback
    # es = EarlyStopping(monitor='val_loss', patience=train_cfg['patience'])
    es = EarlyStopping(monitor='val_loss', patience=10) # hls4ml experiment
    # Learning rate scheduler 
    # ls = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=int(train_cfg['patience']/3))
    ls = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6) # hls4ml experiment
    
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
    
    train_cfg['optimizer']  = tf.keras.optimizers.legacy.Adam(learning_rate=3E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    #train_cfg['optimizer']  = tf.keras.optimizers.Adam(learning_rate=3E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    train_cfg['metrics']    = ["accuracy"]

    # training in scaling or pruning block
    train_cfg['dynamic_train'] = args.pruning_auto or args.scale or args.quantization_auto 


    @strategy_noDec.sole_strategy_BayesianOptimization(strategy_noDec.scaling_model, strategy_noDec.scaling_score, prj_cfg, train_cfg, param_min=0.125)
    def scale_model(model):
        return model
    
    @strategy_noDec.sole_strategy_BayesianOptimization(strategy_noDec.pruning_model, strategy_noDec.pruning_score, prj_cfg, train_cfg)
    def prune_model(model):
        return model


    # @strategy.pruning_pretrain_auto(pruning_cfg, prj_cfg, train_cfg)
    # @strategy.scale_simple_v1(scale_cfg, prj_cfg, train_cfg)
    # @strategy_noDec.scale_prune_BayesianOptimization(scale_cfg, pruning_cfg, prj_cfg, train_cfg)
    @strategy.quantization_auto(prj_cfg, train_cfg, quantization_cfg)
    def prepare_model():
    
        # if args.model in MODEL_POOL:
        #     model_config = MODEL_POOL[args.model]
        # else:
        #     write_line_to_log(log_cfg, "Model is not found")
        #     write_line_to_log(log_cfg, "Select the model below")
        #     write_line_to_log(log_cfg, "exit()")
        #     for key in MODEL_POOL:
        #         print(key)
        #     exit()

        # write_line_to_log(log_cfg, "\n") 
        # write_line_to_log(log_cfg, f"Model name: {args.model}") 
        # write_line_to_log(log_cfg, f"Model name, cnn filters, fc num: {str(model_config)}") 

        # model = model_config[0](input_shape, n_classes, model_config[1])
        # model.summary()
         
        # for layer in model.layers:
        #     if layer.__class__.__name__ in ['Conv2D', 'Dense']:
        #         w = layer.get_weights()[0]
        #         layersize = np.prod(w.shape)
        #         write_line_to_log(log_cfg, "{}: w shape is {}".format(layer.name,str(w.shape))) # 0 = weights, 1 = biases
        #         write_line_to_log(log_cfg, "{}: w size is {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
        #         if (layersize > 4096 and args.synth): # assuming that shape[0] is batch, i.e., 'None'
        #         #if (layersize > 8192): # assuming that shape[0] is batch, i.e., 'None'
        #             write_line_to_log(log_cfg, "Layer {} is too large ({}) for vivado HLS".format(layer.name,layersize))
        #             write_line_to_log(log_cfg, "exit()")
        #             exit()
        model = tf.keras.models.load_model(f'/mnt/ccnas2/bdp/az321/MetaML-dev/strategies/metaml-v1/projects/{dataset_name}/{args.prev_prj_name}/final_keras_model.tf')
        return model
    

    # train

    train = args.train # True if you want to retrain, false if you want to load a previsously trained model
    n_epochs = args.epochs
    patience = 50


    if train:
     
        print("model after strategy: ")
        model = prepare_model()
        
        # if scale_cfg['enable']:
        #     model = scale_model(model)

        # if pruning_cfg['enable']:
        #     model = prune_model(model)
        model.summary()
        if args.model != 'ResNet18':
            LOSS        = tf.keras.losses.CategoricalCrossentropy()
            OPTIMIZER   = tf.keras.optimizers.legacy.Adam(learning_rate=3E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
            model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])

        history = model.fit(X_train, 
                    Y_train,
                    #batch_size=args.batch,
                    epochs = n_epochs,
                    validation_split=0.1,
                    callbacks = train_cfg['callbacks'] # MY
                    )   
        
        # print(history.history.keys())
        write_line_to_log(log_cfg, "\n\n")
        write_line_to_log(log_cfg, "training history:")
        write_dic_to_log(log_cfg, history.history)

        if pruning_cfg['enable']:
            model = strip_pruning(model)

    else:
        if pruning_en:
            co = {}
            _add_supported_quantized_objects(co)
            co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
            model = tf.keras.models.load_model(f'{directory}/final_keras_model.tf', custom_objects=co)
            model  = strip_pruning(model)

        else:
            model = tf.keras.models.load_model(f'{directory}/final_keras_model.tf')
            # model = tf.keras.models.load_model(train_cfg['model_final_loc'])
        
        model.summary()
        
    model.save(f'{directory}/final_keras_model.tf')
     
    start_hls4ml_time = time.strftime("%Y%m%d-%H%M%S")
    write_line_to_log(log_cfg, "\n") 
    write_line_to_log(log_cfg, f"start hls4ml time: {start_hls4ml_time}") 

#hls4ml 
    # if pruning_en:
    #    model  = strip_pruning(model)
    
    if(args.model=='ResNet8' or args.model == 'LHC_CNN'):
        hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
        hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
        hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    
    #First, the baseline model
    #hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='ap_fixed<16,6>')
    default_precision = 'ap_fixed<18,8>'
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision=default_precision)
    
    # Set the precision and reuse factor for the full model
    #hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config['Model']['ReuseFactor'] = args.reuse_factor
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

    # # load baseline model (QModel train 10 epoch)
    # classes = ['g', 'q', 't', 'w', 'z']

    # fig, ax = plt.subplots(figsize=(9, 9))
    # plotting.makeRoc(Y_test, y_keras, classes, linestyle='-')
    # plotting.makeRoc(Y_test, y_hls, classes,  linestyle='--')

    # from matplotlib.lines import Line2D
    # from matplotlib.legend import Legend
    # lines = [Line2D([0], [0], ls='-'), Line2D([0], [0], ls='--')]
    # leg = Legend(ax, lines, labels=['keras', 'hls4ml'], loc='lower right', frameon=False)

    # ax.add_artist(leg)

    # plt.savefig("figure_ROC.png")
    
    
    start_synth_time = time.strftime("%Y%m%d-%H%M%S")
    write_line_to_log(log_cfg, "\n") 
    write_line_to_log(log_cfg, f"start synthesize time: {start_synth_time}") 

    # check_sparsity(model, pruning_en)
    
    synth = args.synth # Only if you want to synthesize the models yourself (>1h per model) rather than look at the provided reports.
    if synth:
        hls_model.build(csim=False, synth=True, vsynth=True)

        hls4ml.report.read_vivado_report(f'{directory}/{project_name}')
    
        # data_pruned_ref = getReports(f'{directory}/{project_name}')
        # print(f"get report: {data_pruned_ref}")
    
        # print(f"\n Resource usage and latency: {project_name}")
        # pprint.pprint(data_pruned_ref)
        # write_dic_to_log(log_cfg, data_pruned_ref)
    # -
    
    end_time = time.strftime("%Y%m%d-%H%M%S")
    write_line_to_log(log_cfg, "\n") 
    write_line_to_log(log_cfg, f"End time: {end_time}") 


