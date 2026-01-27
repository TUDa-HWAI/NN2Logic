# NN2Logic

## Installation

Required: environment variable `GUROBI_HOME` pointing to the installation of Gurobi. Installed boost libraries.
Expects the risc-v compiler toolchain binaries to be installed into `/opt/riscv/bin`.


```bash
$ mkdir build
$ cd build
$ cmake -G Ninja ..
$ ninja install  # installs the python library into the project root
```

## Usage

See `qat-occupancy` for an example.

```bash
$ python modelQ.py  # quantizes the model provided by occupancy.ckpt
                    # and uses QTreeBuilder to do the Decision Tree Conversion
$ python hybrid.py  # takes the qat-w-samples.json (produced by the previous command) 
                    # and creates the hybrid implementation as well as the reference.
```