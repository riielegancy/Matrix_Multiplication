
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")


ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif


DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))


CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif



NVCC            ?= $(shell which nvcc > /dev/null && echo "nvcc" || echo "$(CUDA_BIN_PATH)/nvcc")
CC 				=  g++


EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
#CFLAGS = -O3 -Wall std=gnu99
CFLAGS=-O3 -Wall


GENCODE_SM10    := -gencode arch=compute_10,code=sm_10 -Wno-deprecated-gpu-targets
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20 -Wno-deprecated-gpu-targets
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM20)



ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif


ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif


INCLUDES      := -I$(CUDA_INC_PATH) -I. -I.. -I../../common/inc


CL_LIBS=OpenCL


SRCS=$(wildcard *.c)
CPP_SRCS=$(wildcard *.cpp)
CU_SRCS=$(wildcard *.cu)
OBJS=$(SRCS:.c=.o) $(CPP_SRCS:.cpp=.o)  $(CU_SRCS:.cu=.o)
EXE=matrixmul



$(EXE):$(OBJS)
	$(CC) $(INCLUDES) $(OBJS) -L $(CUDA_LIB_PATH) -l$(CL_LIBS) $(LDFLAGS) $(EXTRA_LDFLAGS) -o $(EXE)

%.o:%.cpp
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

%.o:%.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJS) $(CU_OBJS) $(EXE)
