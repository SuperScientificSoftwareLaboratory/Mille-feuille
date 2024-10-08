# Mille-feuille
A Tile-Grained Mixed Precision Single-Kernel Conjugate Gradient Solver on GPUs
## Paper
This is the code of our paper published at SC '24:
Dechuang Yang, Yuxuan Zhao, Yiduo Niu, Weile Jia, En Shao, Weifeng Liu, Guangming Tan and Zhou Jin. Mille-feuille: A Tile-Grained Mixed Precision Single-Kernel Conjugate Gradient Solver on GPUs. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '24). https://www.ssslab.cn/assets/papers/2024-yang-Millefeuille.pdf
## Introduction
Conjugate gradient (CG) and biconjugate gradient stabilized (BiCGSTAB) are effective methods used for solving sparse linear systems. We in this paper propose Mille-feuille, a
new solver for accelerating CG and BiCGSTAB on GPUs. We first analyze the two methods and list three findings related to the use of mixed precision, the reduction of kernel synchronization costs, and the awareness of partial convergence during the iteration steps. Then, (1) to enable tile-grained mixed precision, we develop a tiled sparse format; (2) to reduce synchronization costs, we leverage atomic operations that make the whole solving procedure work within a single GPU kernel; (3) to support a partial convergence-aware mixed precision strategy, we enable tile-wise on-chip dynamic precision conversion within the single kernel at runtime.
## Installation
Mille-feuille evaluation requires an NVIDIA GPU with compute capability 8.0 and the CUDA GPU driver and the nvcc CUDA compiler, all of them are included with the CUDA Toolkit.
## Execution
Our test programs currently support input files encoded using the matrix market format. All matrix market datasets used in this evaluation are publicly available from the SuiteSparse Matrix Collection.

1. The command 'make xxx' generates an executable file.

```
make cg
make bicg
```
2. Run code on matrix data. Running the program requires one parameter: matrix path.

```
./main-cg matrix.mtx
./main-bicg matrix.mtx
```

3. Example

```
cd test
sh run_cg.sh
sh run_bicg.sh
```

4. The Performance result will appear in the data folder.

## Contact us
If you have any questions about running the code, please contact Dechuang Yang.

E-mail: dechuang.yang@student.cup.edu.cn.
