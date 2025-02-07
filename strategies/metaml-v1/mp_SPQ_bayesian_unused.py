import numpy as np
import time
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score

import hls4ml
# import plotting

import tensorflow.compat.v2 as tf

import os
import sys
os.environ['PATH'] = '/mnt/ccnas2/bdp/opt/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from metaml.v1 import *
from metaml.v1.ah4ml import *


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory (dir1)
project_root = os.path.dirname(current_dir)

# Add hls_path to the Python path
hls_path = os.path.join(project_root, 'hls_strategy')
sys.path.append(hls_path)

import pdb

import hls_strategy 
from balance import *
import quantize as hls_quantize
import hls_utils
from history import log_accuracy, load_history, save_history, parse_report


from bayes_opt import BayesianOptimization as BOptimize
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-PJN", "--project_name", type=str, default="meta_ml", help="project name")
parser.add_argument("-DN", "--dataset_name", type=str, default="svhn",    help="dataset name")
parser.add_argument("-EP" , "--epochs", type=int,   default=1, help="number of epochs")
parser.add_argument("-T",   "--train",  type=int,   default=1, help="train?")
parser.add_argument(        "--batch_size", type=int,  default=32, help="batch size")
parser.add_argument("-LOG", "--log",    type=int,   default=1, help="log")
parser.add_argument("-SYN", "--synth",  type=int,   default=0, help="Synthesis?")
parser.add_argument("-FPGA", "--fpga_name", type=str, default="U250", help="target fpga name" )
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

parser.add_argument("--test_acc", type=int, default=0, help="test acc on the full dataset" )
parser.add_argument("-QNT", "--quant",    type=int,default=0, help="hls quant enable ")

parser.add_argument(        "--max_accuracy_loss",    type=float,default=0.04, help="hls quant max accuracy loss")
parser.add_argument("-G", "--gpu",    type=int,default=0, help="gpu id")



args = parser.parse_args()


if __name__ == "__main__":
    
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)


    project_name = args.project_name
    dataset_name = args.dataset_name 
    fpga_name       = args.fpga_name
    fpga_part       = get_fpga_part(fpga_name) 
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
    # scale_en = args.scale 

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


    gpu_id = args.gpu
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_visible_devices(gpu_devices[gpu_id], 'GPU')
        write_line_to_log(log_cfg, f"using gpu {gpu_id}") 
    # print("Visible GPUs:", tf.config.experimental.get_visible_devices('GPU'))
    # pdb.set_trace()

    # create csv file to record data
    csv_file_path = f"{directory}/data.csv"
    if not os.path.exists(csv_file_path):
        data = {
            "iteration_total": [],
            "alpha_s": [],
            "alpha_p": [],
            "alpha_q": [],
            "accuracy": [],
            "BRAM": [],
            "DSP": [],
            "FF": [],
            "LUT": [],
            "score": []
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file_path, index=False)
    
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


    # pruning configuration
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


    # scaling configuration
    scale_cfg = {}
    scale_cfg['enable']        = 0
    scale_cfg['max_trials']    = args.scaling_trial
    scale_cfg['auto']          = args.scaling_auto
    scale_cfg['step']          = args.scaling_step # rate = 1 / step
    scale_cfg['rate']          = args.scaling_rate # rate = 1 / step
    scale_cfg['patience']      = 3
    # scale_cfg['threshold']     = args.accuracy_threshold

    train_cfg = {}
    train_cfg['enable']     = args.train
    train_cfg['n_epochs']   = args.epochs
    train_cfg['batch_size'] = args.batch_size
      
    train_cfg['patience']   = 50
    train_cfg['pruning_en'] = pruning_en
    # train_cfg['scale_en']   = scale_en
    train_cfg['val_split']  = 0.1
    train_cfg['model_chk_loc']  = f'{directory}/{args.model}_chk_baseline_model.h5'
    train_cfg['model_final_loc']  = f'{directory}/{args.model}_final_baseline_model.h5'
    train_cfg['model_weights_loc']  = f'{directory}/{args.model}_final_weights_model.h5'
    train_cfg['model_loc']  = f'{directory}/{args.model}_model'

    # train_cfg['acc_threshold'] = args.accuracy_threshold
    # train_cfg['bayesian_iter'] = args.bayesian_iter

    write_line_to_log(log_cfg, "part of training config")
    write_dic_to_log(log_cfg, train_cfg)


    es = EarlyStopping(monitor='val_loss', patience=10) # hls4ml experiment
    ls = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6) # hls4ml experiment
    
    # model checkpoint callback
    chkp = ModelCheckpoint(train_cfg['model_chk_loc'],
                monitor='val_loss', verbose=1, save_best_only=True, 
                save_weights_only=False, mode='auto', 
                save_freq='epoch')
    prn = pruning_callbacks.UpdatePruningStep()

    
    
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

    train_cfg['bayes_iter'] = 0

    weights = {
        "accuracy": 10,
        "BRAM": 0,
        "DSP": 3,
        "FF": 0,
        "LUT": 1
    }

    benchmark = {
        "accuracy": 0.772,
        "BRAM": 4,
        "DSP": 100,
        "FF": 7919,
        "LUT": 15000
    }


    @strategy.pruning_pretrain_auto(pruning_cfg, prj_cfg, train_cfg)
    @strategy.scale_simple_v1(scale_cfg, prj_cfg, train_cfg)
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

    def black_box_function(prune_acc_loss, scale_acc_loss, quant_acc_loss): 
        bayes_iter = train_cfg['bayes_iter']
        # pipeline = int(pipeline) # pipeline: [0, 1, ..., 6]
        # pipeline_options = ["Q", "SP", "SQ", "PQ", "SPQ", "S", "P"]
        # pipeline_selection = pipeline_options[pipeline]

        write_line_to_log(log_cfg, "\n")
        # write_line_to_log(log_cfg, f"iteration {bayes_iter} with pipeline={pipeline_selection}")
        write_line_to_log(log_cfg, f"iteration {bayes_iter}")
        write_line_to_log(log_cfg, f"prune_acc_loss={str(prune_acc_loss)}")
        write_line_to_log(log_cfg, f"scale_acc_loss={str(scale_acc_loss)}")
        write_line_to_log(log_cfg, f"quant_acc_loss={str(quant_acc_loss)}")
        

        pipeline_en = {}
        pipeline_en["scaling"] = 1
        pipeline_en["pruning"] = 1
        pipeline_en["hls_quantization"] = 1
        # if "S" in pipeline_selection:
        #     pipeline_en["scaling"] = 1
        #     write_line_to_log(log_cfg, f"scaling is enable")
        # else:
        #     pipeline_en["scaling"] = 0
        
        # if "P" in pipeline_selection:
        #     pipeline_en["pruning"] = 1
        #     write_line_to_log(log_cfg, f"pruning is enable")
        # else:
        #     pipeline_en["pruning"] = 0
        
        # if "Q" in pipeline_selection:
        #     pipeline_en["hls_quantization"] = 1
        #     write_line_to_log(log_cfg, f"hls_quantization is enable")
        # else:
        #     pipeline_en["hls_quantization"] = 0
        

        # configuration
        pruning_cfg['acc_delta_ths'] = -prune_acc_loss  # negative, maximum acc loss due to pruning
        pruning_cfg['enable'] = pipeline_en["pruning"]
        if pipeline_en["pruning"]:
            train_cfg['callbacks']  = [ ls, prn ]
        else:
            train_cfg['callbacks']  = [ es, ls]

        scale_cfg['enable'] = pipeline_en["scaling"]
        scale_cfg['threshold'] = scale_acc_loss

        model = prepare_model()
        
        model.summary()

        if pruning_cfg['enable']:
            model = strip_pruning(model)
            
        model.save(f'{directory}/final_keras_model_iter{bayes_iter}.tf')
        train_cfg['bayes_iter'] += 1
        
        start_hls4ml_time = time.strftime("%Y%m%d-%H%M%S")
        write_line_to_log(log_cfg, "\n") 
        write_line_to_log(log_cfg, f"start hls4ml time: {start_hls4ml_time}") 

    #hls4ml 
        
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
        hls_config['Model']['ReuseFactor'] = 1
        hls_config['Model']['Strategy'] = 'Latency'
        hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'
        
        # plotting.print_dict(hls_config)

        #exit()
        
        cfg = hls4ml.converters.create_config(backend='Vivado')
        cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
        cfg['HLSConfig']  = hls_config
        cfg['KerasModel'] = model
        cfg['OutputDir']  = f'{directory}/{project_name}_iter{bayes_iter}'
        cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

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
        

    # hls quantization
        # Add recompile method to HLSModel class
        type(hls_model).recompile = recompile
        
        quant_cfg = {}
        # if args.quant !=0:
        #     quant_cfg['en'] = True
        # else:
        #     quant_cfg['en'] = False
        quant_cfg['en'] = pipeline_en["hls_quantization"]
            
        quant_cfg['max_accuracy_loss'] = quant_acc_loss
        #quant_cfg['default_precision'] = default_precision

        synth_cfg = {}
        if args.synth !=0:
            synth_cfg['en'] = True
        else:
            synth_cfg['en'] = False

        test_acc_cfg = {}
        if args.test_acc !=0:
            test_acc_cfg['en'] = True
        else:
            test_acc_cfg['en'] = False

        
        test_data_full = tuple([X_test, Y_test]) 
        prj_cfg['hls_model']    = hls_model
        prj_cfg['krs_model']    = model
        prj_cfg['test_dataset_full'] = test_data_full
        prj_cfg['test_dataset_quant'] = test_data_full
        prj_cfg['n_samples']    = 3000      # 3000 
        prj_cfg['max_accuracy_loss'] = 0.01
        prj_cfg['fpga_part']    = TARGET_DEVICE
        prj_cfg['log_enable']   = True


        # @hls_strategy.synth_hls(synth_cfg, prj_cfg)
        # @hls_strategy.test_acc_hls(test_acc_cfg, prj_cfg)
        @hls_strategy.quant_hls(quant_cfg, prj_cfg) 
        def prepare_firmware ():
            firmware = Firmware(f'{directory}/{project_name}_iter{bayes_iter}')
            return firmware

        firmware = prepare_firmware()

        test_data = (np.ascontiguousarray(X_test), Y_test)
        accuracy_hls = hls_quantize.get_accuracy( 
            firmware    = firmware,
            hls_model   = prj_cfg['hls_model'], 
            test_data   = test_data 
        )
        
        write_line_to_log(log_cfg, "After HLS quantization:")
        write_line_to_log(log_cfg, f'HLS4ML accuracy = {accuracy_hls}')

        hls_utils.synth_firmware(firmware)
        report = parse_report(f'{directory}/{project_name}_iter{bayes_iter}')
        report = report['total']
        # print(report)

        def computeImprovement (usage, bm): 
            return (float(bm) - float(usage)) / float(bm)
        
        data = report['BRAM_18K']
        write_line_to_log(log_cfg, f'BRAM_18K = {data}')
        data = report['DSP48E']
        write_line_to_log(log_cfg, f'DSP48E = {data}')
        data = report['FF']
        write_line_to_log(log_cfg, f'FF = {data}')
        data = report['LUT']
        write_line_to_log(log_cfg, f'LUT = {data}')

        final_score = accuracy_hls
        final_score += computeImprovement(report['BRAM_18K'], benchmark['BRAM']) * weights['BRAM']
        final_score += computeImprovement(report['DSP48E'], benchmark['DSP']) * weights['DSP']
        final_score += computeImprovement(report['FF'], benchmark['FF']) * weights['FF']
        final_score += computeImprovement(report['LUT'], benchmark['LUT']) * weights['LUT']
        
        write_line_to_log(log_cfg, f'Final score = {final_score}')

        if accuracy_hls < 0.6:
            final_score = -1000

        # save data
        df = pd.read_csv(csv_file_path)
        # Add a new row to the DataFrame
        new_row = {
            "iteration_total": bayes_iter,
            "alpha_s": scale_acc_loss,
            "alpha_p": prune_acc_loss,
            "alpha_q": quant_acc_loss,
            "accuracy": accuracy_hls,
            "BRAM": report['BRAM_18K'],
            "DSP": report['DSP48E'],
            "FF": report['FF'],
            "LUT": report['LUT'],
            "score": final_score
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Save the updated DataFrame back to the CSV file
        df.to_csv(csv_file_path, index=False)

        return final_score
    
    params_nn ={
        'prune_acc_loss': (0.01, 0.04),
        'scale_acc_loss': (0.01, 0.04),
        'quant_acc_loss': (0.01, 0.04),
        # 'pipeline': (0, 6.99)  # {S, P, Q, SP, SQ, PQ, SPQ}
    }
    
    optimizer = BOptimize(
        f=black_box_function,
        pbounds=params_nn,
        random_state=1,
        allow_duplicate_points=True
    )

    bayes_log_path = f'{directory}/logs_Bayesian.log'
    if os.path.exists(bayes_log_path):
        load_logs(optimizer, logs=[bayes_log_path])
        init_len = len(optimizer.space)
        train_cfg['bayes_iter'] = init_len
        write_line_to_log(log_cfg, f"Initial Bayesian iteration number is {init_len}") 
    
    logger = JSONLogger(path=bayes_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    

    # optimizer.maximize(init_points=2, n_iter=5)  # TODO: change iteration to config parameter
    optimizer.maximize(init_points=2, n_iter=20)
    params_nn_ = optimizer.max['params']

    write_line_to_log(log_cfg, f"Bayesian Optimization final prune_acc_loss={str(params_nn_['prune_acc_loss'])}, quant_acc_loss={str(params_nn_['quant_acc_loss'])}")
    
    
    end_time = time.strftime("%Y%m%d-%H%M%S")
    write_line_to_log(log_cfg, "\n") 
    write_line_to_log(log_cfg, f"End time: {end_time}") 
