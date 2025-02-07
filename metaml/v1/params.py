

from .models.VGG import *
from .models.LeNet import *
from .models.ResNet import *
from .models.LHC_DNN import *


# pool of model, [0] is reserved for future parameters
# model_name, [filters of CNNs, out of FCs ]
MODEL_POOL = {
'VGG6'  : (VGG6,  [0]),
#'VGG7'  : (VGG7_v8192,  [16, 32]), # 8192
#'VGG7'  : (VGG7_v4096,  [16, 32]), # 4096
'VGG7'  : (VGG7_v4096,  [14, 28]), # 4096
'VGG11' : (VGG11, [16, 64]),

'LeNet5'    : (LeNet5_v4096,  [12, 4]),

#'ResNet8'   : (ResNet8_v8192, [16]), # 8192
'ResNet8'   : (ResNet8_v2_4096, [16]), # 4096
'ResNet9'   : (ResNet9_4096, [16]), # 4096
'ResNet10'  : (ResNet10,[4]),
'ResNet18'  : (ResNet18,[64]),

# models For Large Hadron Collider, Level 1 trigger
 'LHC_DNN'  : (LHC_DNN, [64]),
 #'LHC_DNN'  : (LHC_DNN, [8]),
 'LHC_CNN'  : (LHC_CNN, [8, 32]),

}

DATASET_POOL = {
'mnist'     : 1,
'svhn'      : 1,
'cifar10'   : 1,
'jets_hlf'  : 1,
}
