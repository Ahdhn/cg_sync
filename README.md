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
| Threads            |  Blocks            | Size               | Baseline (ms) | CGSync (ms) |
|--------------------|--------------------|--------------------|---------------|-------------|
| 32                 | 1344               | 43008              |  4.89674      | 9.16368     |
| 64                 | 1344               | 86016              |  5.81734      | 10.8401     |
| 128                | 1008               | 129024             |  5.11898      | 7.32672     |
| 256                | 504                | 129024             |  4.15232      | 5.76922     |
| 512                | 252                | 129024             |  4.51789      | 4.7063      |
| 1024               | 84                 | 86016              |  3.61882      | 3.92192     |


## Ubuntu - GCC 9.4 - CUDA 11.8 - A100
| Threads            | Blocks             | Size               | Baseline (ms) | CGSync (ms)| 
|--------------------|--------------------|--------------------|---------------|------------|
| 32                 | 3456               | 110592             | 10.382        | 37.427     |       
| 64                 | 3456               | 221184             | 10.1224       | 37.6356    |       
| 128                | 1728               | 221184             | 8.51709       | 20.8437    |       
| 256                | 864                | 221184             | 7.69366       | 12.2465    |       
| 512                | 432                | 221184             | 7.41354       | 9.90435    |       
| 1024               | 216                | 221184             | 3.79459       | 5.4601     | 


## Windows - VS2022 - CUDA 12.1 -  GTX 1050
| Threads             | Blocks              | Size             | Baseline (ms) | CGSync (ms)|
|---------------------|---------------------|------------------|---------------|------------|
| 32                  | 160                 | 5120             | 3.23789       | 4.25165    |
| 64                  | 160                 | 10240            | 3.44758       | 5.29715    |
| 128                 | 80                  | 10240            | 4.51174       | 4.69914    |
| 256                 | 40                  | 10240            | 3.4568        | 4.15539    |
| 512                 | 20                  | 10240            | 2.57024       | 4.58138    |
| 1024                | 10                  | 10240            | 3.34541       | 3.74864    |
                                                                                            