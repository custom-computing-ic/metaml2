# from pathlib import Path
# import numpy as np
# import pprint 
import hls4ml



import argparse

def process_arguments():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-D', '--dataset', type=str, default='svhn')
    parser.add_argument('-P', '--project', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = process_arguments()
    dataset = args.dataset
    project = args.project

    hls4ml.report.read_vivado_report(f'projects/{dataset}/{project}/{project}')
