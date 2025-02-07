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


parser = argparse.ArgumentParser()
parser.add_argument("-PJN", "--project_name", type=str, default="prj6_synth", help="project name")
parser.add_argument("-DN", "--dataset_name", type=str, default="jets_hlf",    help="dataset name")
parser.add_argument("-LOG", "--log",    type=int,   default=1, help="log")
parser.add_argument("-SYN", "--synth",  type=int,   default=0, help="Synthesis?")
parser.add_argument("-FPGA", "--fpga_name", type=str, default="U250", help="target fpga name" )
parser.add_argument("-I", "--iteration", type=int, default=0, help="iteration number to be synth" )
parser.add_argument(        "--prev_project_name", type=str, default="prj6_keras_hls_combine_debug", help="project name")

args = parser.parse_args()


if __name__ == "__main__":
    
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    project_name = args.project_name
    dataset_name = args.dataset_name 
    prev_project_name = args.prev_project_name
    fpga_name       = args.fpga_name
    fpga_part       = get_fpga_part(fpga_name) 
    log = args.log
    log_file_name = f"log_{project_name}.txt"
    directory = f"projects/{dataset_name}/{project_name}"
    synth_dir = f"projects/{dataset_name}/{prev_project_name}"
    
    prj_cfg = {}
    prj_cfg['log']              = {}
    prj_cfg['log']['enable']    = log
    prj_cfg['log']['log_file']  =  f'{directory}/{log_file_name}'
    log_cfg = prj_cfg['log'] 
    os.system(f"mkdir -p {directory}")
    iter = args.iteration
    # write_line_to_log(log_cfg, f'directory is {prev_project_name}_iter{iter}') 
    

    # start_hls4ml_time = time.strftime("%Y%m%d-%H%M%S")
    # write_line_to_log(log_cfg, "\n") 
    # write_line_to_log(log_cfg, f"start hls4ml time: {start_hls4ml_time}") 
    
    # firmware = Firmware(f'{synth_dir}/{prev_project_name}_iter{iter}')

    # start_synth_time = time.strftime("%Y%m%d-%H%M%S")
    # write_line_to_log(log_cfg, "\n") 
    # write_line_to_log(log_cfg, f"start synthesize time: {start_synth_time}") 

    synth = args.synth # Only if you want to synthesize the models yourself (>1h per model) rather than look at the provided reports.
    if synth:
        # hls_utils.synth_firmware(firmware)
        report = parse_report(f'{synth_dir}/{prev_project_name}_iter{iter}')
        print(report)
        # report = report['total']
        # data = report['average_case_latency']
        # write_line_to_log(log_cfg, f'latency:{data}') 
        # data = report['BRAM_18K']
        # write_line_to_log(log_cfg, f'BRAM_18K:{data}') 
        # data = report['DSP48E']
        # write_line_to_log(log_cfg, f'DSP48E:{data}') 
        # data = report['FF']
        # write_line_to_log(log_cfg, f'FF:{data}') 
        # data = report['LUT']
        # write_line_to_log(log_cfg, f'LUT:{data}')

    # end_time = time.strftime("%Y%m%d-%H%M%S")
    # write_line_to_log(log_cfg, "\n") 
    # write_line_to_log(log_cfg, f"End time: {end_time}") 

    
