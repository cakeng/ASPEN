TARGET=main
ALIB=libaspen.a
OBJECTS=build_info.o apu.o apu_nasm.o apu_file_io.o input_parser.o darknet_parser.o util.o 
OBJECTS+=rpool.o dse.o naive_kernels.o tiled_kernels.o avx2_kernels.o neon_kernels.o
AVX2=0
NEON=1
ANDROID=0
DEBUG=0
SUPPRESS_OUTPUT=0

CC=gcc

OBJDIR=./obj/
VPATH=./src 
DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

BUILD_INFO_TIME = $(shell LC_TIME=en_US date)
BUILD_INFO_GCC = $(shell gcc --version | grep -Ei "gcc \([0-9a-zA-Z\. -~]+\) [0-9\.]+")
BUILD_INFO_UNAME = $(shell uname -srvpim)
BUILD_INFO_BRANCH = $(shell git log -1 | grep -Eio "commit [0-9a-zA-Z]+")

OPTS= -Ofast -march=native -funroll-loops
LDFLAGS=-lm -lgomp
COMMON= -Iinclude/
ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -DDEBUG
endif
CFLAGS= -Wall -Wno-unused-variable -fopenmp
ARFLAGS=rcs
ifeq ($(AVX2), 1)
OPTS+=-mavx2 -mfma -DAVX2 -m64
endif
ifeq ($(NEON), 1)
OPTS+=-DNEON
endif

ifeq ($(ANDROID), 1) 
#NDK Config
TOOLCHAIN=$(NDK)/toolchains/llvm/prebuilt/linux-x86_64
CC_TARGET=aarch64-linux-android
API=29

CC=$(TOOLCHAIN)/bin/$(CC_TARGET)$(API)-clang
AR=$(TOOLCHAIN)/bin/llvm-ar
OPTS=-Ofast -funroll-loops -DNEON
LDFLAGS=-lm -fopenmp -pthread -landroid -llog --sysroot=$(TOOLCHAIN)/sysroot
CFLAGS=-Wall -fPIC -fopenmp -static-openmp
BUILD_INFO_GCC = $(shell $(CC) --version | tr '\n' ' ' | grep -Eio "Android .+ posix")
endif

ifeq ($(SUPPRESS_OUTPUT), 1) 
OPTS+=-D_SUPPRESS_OUTPUT
endif
INFO_FLAGS= -DBUILD_INFO_TIME="\"$(BUILD_INFO_TIME)"\" -DBUILD_INFO_GCC="\"$(BUILD_INFO_GCC)\"" -DBUILD_INFO_UNAME="\"$(BUILD_INFO_UNAME)\"" -DBUILD_INFO_BRANCH="\"$(BUILD_INFO_BRANCH)\""
INFO_FLAGS+= -DBUILD_INFO_FLAGS="\"$(COMMON) $(LDFLAGS) $(CFLAGS) $(OPTS)"\"
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

obj:
	mkdir -p obj

clean:
	rm -rf $(ALIB) $(OBJS) $(OBJDIR) $(EXEOBJS) $(TARGET)

