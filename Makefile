TARGET=main
ALIB=libasapen.a
OBJECTS=build_info.o apu.o apu_file_io.o input_parser.o darknet_parser.o util.o
AVX2=1
NEON=0
GPU=0
DEBUG=1
SUPPRESS_OUTPUT=0

CC=gcc
NVCC=nvcc

OBJDIR=./obj/
VPATH=./src 
DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

BUILD_INFO_TIME = $(shell LC_TIME=en_US date)
BUILD_INFO_GCC = $(shell gcc --version | grep -Ei "gcc \([0-9a-zA-Z\. -~]+\) [0-9\.]+")
BUILD_INFO_UNAME = $(shell uname -srvpim)
BUILD_INFO_BRANCH = $(shell git log -1 | grep -Eio "commit [0-9a-zA-Z]+")
BUILD_INFO_NVCC := $(shell nvcc --version | grep -Eoi "Build .*")

CFLAGS= -Wall -fopenmp
ARFLAGS=rcs
LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lopenblas -lgomp
COMMON=-I/usr/local/cuda/include/ -Iinclude/
ARCH= 	-gencode arch=compute_80,code=[sm_80,compute_80] \
		-gencode arch=compute_86,code=[sm_86,compute_86]
OPTS=-Ofast
ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -DDEBUG
endif
ifeq ($(SUPPRESS_OUTPUT), 1) 
OPTS+=-D_SUPPRESS_OUTPUT
endif
INFO_FLAGS= -DBUILD_INFO_TIME="\"$(BUILD_INFO_TIME)"\" -DBUILD_INFO_GCC="\"$(BUILD_INFO_GCC)\"" -DBUILD_INFO_UNAME="\"$(BUILD_INFO_UNAME)\"" -DBUILD_INFO_BRANCH="\"$(BUILD_INFO_BRANCH)\""
INFO_FLAGS+= -DBUILD_INFO_FLAGS="\"$(COMMON) $(LDFLAGS) $(CFLAGS) $(OPTS)"\"
INFO_FLAGS+= -DBUILD_INFO_NVCC="\"$(BUILD_INFO_NVCC)\""
INFO_FLAGS+= -DBUILD_INFO_GPU_ARCH="\"$(ARCH)\""
OBJS= $(addprefix $(OBJDIR), $(OBJECTS))
EXEOBJSA= $(addsuffix .o, $(TARGET))
EXEOBJS= $(addprefix $(OBJDIR), $(EXEOBJSA))

all: obj $(TARGET)

$(TARGET): $(EXEOBJS) $(ALIB) 
	$(CC) $(COMMON) $(CFLAGS) $(OPTS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(OBJDIR)build_info.o: build_info.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) $(INFO_FLAGS) $(OPTS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) $(OPTS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

clean:
	rm -rf $(TARGET) $(OBJS)

