# CGSync [![Windows](https://github.com/Ahdhn/cg_sync/actions/workflows/Windows.yml/badge.svg)](https://github.com/Ahdhn/cg_sync/actions/workflows/Windows.yml) [![Ubuntu](https://github.com/Ahdhn/cg_sync/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/Ahdhn/cg_sync/actions/workflows/Ubuntu.yml)

Experimenting with global sync across blocks. 


## Build 
```
mkdir build
cd build 
cmake ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 


# Results
## Windows - VS2019 - CUDA 11.4 - RTX A6000

|Threads            | Blocks             | Baseline (ms)      | CGSync (ms) |
|-------------------|--------------------|--------------------|-------------|                                  
|32                 | 1344               | 4.77987            | 9.07264     |
|64                 | 1344               | 5.57978            | 10.6168     |
|128                | 1008               | 4.49536            | 7.27757     |
|256                | 504                | 4.11648            | 5.28179     |
|512                | 252                | 4.64589            | 6.00253     | 
|1024               | 84                 | 2.82522            | 3.94544     |