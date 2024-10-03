CUDA_INSTALL_PATH=/usr/local/cuda-12.0
NVCCFLAGS=-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -fpermissive
LDFLAGS= -L $(CUDA_INSTALL_PATH)/lib64 -g 
INCLUDES = -I $(CUDA_INSTALL_PATH)/include
.PHONY :cg bicg
cg:
	nvcc src/main-cg.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o main-cg -Xcompiler -fopenmp -O3 -maxrregcount=32
bicg:
	nvcc src/main-bicg.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o main-bicg -Xcompiler -fopenmp -O3 -maxrregcount=32
clean:
	rm main-cg
	rm main-bicg
	rm data/*.csv