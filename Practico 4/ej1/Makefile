CC = nvcc

VALUE_TYPE = double

NVCC_FLAGS = -Xcompiler -ftree-vectorize -Xcompiler -fopenmp -O3 -w -m64 -rdc=true \
#             -gencode=arch=compute_61,code=sm_61 \
                         -arch=sm_61  \
             -Xptxas -dlcm=cg

CUDA_INSTALL_PATH = /usr/local/cuda
MKLROOT = /local/gpgpu/software/intel/mkl/2021.4.0

INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I./include -I${MKLROOT}/include

MKL_LIBS = -lpthread -lm
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcuda -lcusparse -lnvidia-ml
LIBS = $(CUDA_LIBS) $(MKL_LIBS)

FILE ?= main

make:
        $(CC) $(NVCC_FLAGS) $(FILE).cu -o $(FILE) $(INCLUDES) $(LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__