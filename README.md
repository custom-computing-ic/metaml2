# MetaML v2

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
    git clone git@github.com:custom-computing-ic/metaml2.git
   ```

4. Speed-up anaconda solver (not required but highly recommended) by installing [libnamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community):
   ```bash
   conda update -n base conda
   conda install -n base conda-libmamba-solver
   conda config --set solver libmamba
   ```

5. Create project environment:

   Run `conda.env.build` script:
      ```bash
      bash conda.env.build # artisan or heterograph
      ```
6. Activate project environment:
   ```bash
   conda activate metaml
   ```



