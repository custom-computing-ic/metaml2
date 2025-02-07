# CC Project: DART

## Installation

__Preparation:__

1. Install `Anaconda` or `Miniconda`. To install Miniconda, follow the instructions [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers).

2. Optional dependencies:
   - **GPU:** Install the Nvidia driver. Ensure that nvidia-smi can be invoked successfully. This version of the software has been tested with Nvidia driver version 525.

   - **Vivado HLS:** Install Vivado 2019.2 or 2020.1 (recommended). For Ubuntu, install the following packages before installing Vivado HLS:

      ```bash
      sudo apt install libtinfo5 libncurses5 libx11-6 libc6-dev
      ```
      Once it is installed, you must source `settings64.sh`. We recommend adding the following your `.bashrc`:

      ```bash
      source /opt/Xilinx/Vivado/2020.1/settings64.sh
      ```
3. Clone this repository:
   ```bash
    git clone git@github.com:custom-computing-ic/ccdart.git
   ```

4. Speed-up anaconda solver (not required but highly recommended) by installing [libnamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community):
   ```bash
   conda update -n base conda
   conda install -n base conda-libmamba-solver
   conda config --set solver libmamba
   ```

5. Create project environment:

   The script ``bin/dart`` must be used to create a DART project environment using Anaconda. Running the script without arguments shows the list of available project environments:

   * `heterograph`: a library developed to support heterogeneous graph models
   * `artisan`: a metaprogramming framework for C++ code descriptions
   * `metaml_cpu`: a metaprogramming framework supporting reconfigurable ML models using HLS4ML
   * `metaml`: same as above but with NVIDIA GPU support

   To build an anaconda environment, pass the project name as argument, e.g.:
      ```bash
      bin/dart metaml # artisan or heterograph
      ```
6. Documentation:

   Technical and user (notebook) documentation can be found in: https://custom-computing-ic.github.io/ccdart/.

   Additional material:
   * `artisan`: check `projects/artisan/tutorial` for a set of examples.



