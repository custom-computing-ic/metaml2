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

from metaml.v1 import *

def write_line_to_log(text: str):
    if log:
        with open(f'{directory}/{log_file_name}', 'a') as file:
            file.write(text + "\n")
            print("Write to Log: "+text)

def write_dic_to_log(text_dict):
    if log:
        with open(f'{directory}/{log_file_name}', 'a') as file:

            file.write("\n")
            for key in text_dict:
                file.write(key+":"+str(text_dict[key]) + "\n")
            file.write("\n")

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


def prepare_model():

    input_shape, n_classes = get_info_svhn()

    if args.model in MODEL_POOL:
        model_config = MODEL_POOL[args.model]
    else:
        print("Model is not found")
        print("Select the model below")
        for key in MODEL_POOL:
            print(key)
        exit()

    write_line_to_log("\n")
    write_line_to_log(f"Model name: {args.model}")
    write_line_to_log(f"Model name, cnn filters, fc num: {str(model_config)}")

    model = model_config[0](input_shape, n_classes, model_config[1])
    model.summary()

    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv2D', 'Dense']:
            w = layer.get_weights()[0]
            layersize = np.prod(w.shape)
            write_line_to_log("{}: w shape is {}".format(layer.name,str(w.shape))) # 0 = weights, 1 = biases
            write_line_to_log("{}: w size is {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
            if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
            #if (layersize > 8192): # assuming that shape[0] is batch, i.e., 'None'
                print("Layer {} is too large ({}) for vivado HLS".format(layer.name,layersize))
                exit()

    #exit()
    train = args.train # True if you want to retrain, false if you want to load a previsously trained model
    n_epochs = args.epochs
    patience = 20

    if train:

        LOSS        = tf.keras.losses.CategoricalCrossentropy()
        OPTIMIZER   = tf.keras.optimizers.Adam(learning_rate=3E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)

        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])

        # early stopping callback
        es = EarlyStopping(monitor='val_loss', patience=patience)
        # Learning rate scheduler
        ls = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=int(patience/2))

        # model checkpoint callback
        # this saves our model architecture + parameters into mlp_model.h5
        chkp = ModelCheckpoint(f'{directory}/{args.model}_baseline_model.h5',
                    monitor='val_loss', verbose=1, save_best_only=True,
                    save_weights_only=False, mode='auto',
                    save_freq='epoch')

        callbacks = [ es, ls, chkp ]

        history = model.fit(X_train,
                  Y_train,
                  #batch_size=args.batch,
                  epochs = n_epochs,
                  validation_split=0.3,
                  callbacks = callbacks)

        #print(history.history.keys())
        write_line_to_log("\n\n")
        write_line_to_log("training history:")
        write_dic_to_log(history.history)
        #model.save(f'{directory}/baseline_model.h5')

    else:
        model = tf.keras.models.load_model(f'{directory}/{args.model}_baseline_model.h5')
    return model



parser = argparse.ArgumentParser()
parser.add_argument("-PJN", "--project_name", type=str, default="meta_ml", help="project name")
parser.add_argument("-DTN", "--dataset_name", type=str, default="svhn",    help="dataset name")
parser.add_argument("-EP" , "--epochs", type=int,  default=1, help="number of epochs")
parser.add_argument("-T",   "--train",  type=bool, default=1, help="train?")
parser.add_argument(        "--batch",  type=int, default=512, help="train?")
parser.add_argument("-LOG", "--log",    type=bool, default=1, help="log")
parser.add_argument("-SYN", "--synth",  type=bool, default=0, help="Synthesis?")
parser.add_argument("-M",   "--model",  type=str,  default="VGG6", help="model name ")

args = parser.parse_args()


if __name__ == "__main__":

    project_name = args.project_name
    dataset_name = args.dataset_name
    log = args.log

    log_file_name = f"log_{project_name}.txt"
    directory = f"projects/{dataset_name}/{project_name}"
    os.system(f"mkdir -p {directory}")

    X_train, Y_train, X_test, Y_test = load_train_data_svhn()


    start_time = time.strftime("%Y%m%d-%H%M%S")
    write_line_to_log("\n")
    write_line_to_log(f"Start time: {start_time}")
    write_line_to_log(f"Project name: {project_name}")
    write_line_to_log(f"dataset name: {dataset_name}")
    write_line_to_log(f"trainng data size:{str(X_train.shape)}")

    model = prepare_model()
    model.summary()

    #exit()

    evaluate_res = model.evaluate(X_test, Y_test)
    predict_y    = model.predict(X_test)

    write_line_to_log(f'Keras accuracy = {evaluate_res[1]}')


    # +

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

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = model
    cfg['OutputDir']  = f'{directory}/{project_name}'
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

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

    write_line_to_log(f'Test accuracy with {test_num} samples')
    write_line_to_log(f'Keras  accuracy = {accuracy_keras}')
    write_line_to_log(f'HLS4ML accuracy = {accuracy_hls}')


    synth = args.synth # Only if you want to synthesize the models yourself (>1h per model) rather than look at the provided reports.
    if synth:
        hls_model.build(csim=False, synth=True, vsynth=True)

        data_pruned_ref = getReports(f'{directory}/{project_name}')

        print(f"\n Resource usage and latency: {project_name}")
        pprint.pprint(data_pruned_ref)
        write_dic_to_log(data_pruned_ref)
    # -


