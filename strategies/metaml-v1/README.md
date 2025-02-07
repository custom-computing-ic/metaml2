# MetaML strategies (v1)

(from repo: https://github.com/custom-computing-ic/MetaML-dev)


## Quick usage

Ensure to activate the `metaml` environment:

```bash
conda activate metaml
```

### Strategies (API v1)

A number of strategies have been implemented for `v1`:

* __auto pruning:__

   ```python
   python mp_pruning_auto.py -PJN prj_name_lhc_dnn -M LHC_DNN -DN jets_hlf -PRN 1 -PNR 0.5 -EP 10
   python mp_pruning_auto.py -PJN VGG7_auto_sparsity003 -M VGG7 -DN mnist -PRN 1 -PNR 0.5 -EP 10
   python mp_pruning_auto.py -PJN ResNet8_auto_sparsity001 -M ResNet8 -PRN 1 -PNR 0.5 -EP 20
   ```

* __static pruning:__
   ```python
   python mp_pruning_auto.py -PJN ResNet8_sparsity001 -M ResNet8 -PRN 1 -PNR 0.90 -EP 20 --pruning_auto 0
   ```

* __scaling:__
  ```
   python mp_scale_auto.py -PJN lhc_dnn_scale007 -M LHC_DNN -DN jets_hlf -SCL 1 -EP 10
   ```
* __pruning and scaling:__
   ```
   python mp_scale_pruning_auto.py -PJN lhc_dnn_scale010 -M LHC_DNN -DN jets_hlf -SCL 1 -EP 10 -PRN 1 -PNR 0.75
   ```

* __auto quantization:__
  The following example performs the auto quantization algorithm.
  ```python
   python mp_quantization.py -PJN test_static_scaling -M VGG6 -DN svhn -PRN 0 -EP 10 -SYN 0 -SCL 0 -Q 1
   ```

* __static scaling example:__
  The following example performs the static scaling algorithm with default step=8, acc_threshold=0.02
  ```python
  python mp_quantization.py -PJN test_quantization -M VGG6 -DN svhn -PRN 0 -EP 10 -SYN 0 -SCL 1 -SCA 0
  ```

* __synthesis:__
  ```
  python mp_pruning_auto.py -PJN prj_name_lhc_dnn    -M LHC_DNN -DN jets_hlf -PRN 1 -PNR 0.5 -EP 10 -SYN 1
  ```
  or
  ```
  python mp_synth.py -PJN jet_synth_try001 -DN jets_hlf -LD prj_backup/sparisty/lhc_dnn_auto_pruning/LHC_DNN_model_pruned_s0.h5 -IO p -PRN 1
  ```



