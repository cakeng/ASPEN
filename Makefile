TARGET=aspen_benchmark
GPU=1

CC=gcc
NVCC=nvcc

OBJDIR=./obj/
VPATH=./src 
DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

OPTS=-Ofast -march=native -funroll-loops
LDFLAGS= -L. -laspen -lm -lgomp 
COMMON= -I. 
CFLAGS= -fopenmp

ifeq ($(GPU), 1)
LDFLAGS+=-L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas
COMMON+=-I/usr/local/cuda/include/
endif

all: obj $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(COMMON) $(CFLAGS) $(OPTS) $^ -o $@ $(LDFLAGS) $(ALIB)

obj:
	mkdir -p obj

clean:
	rm -rf $(TARGET) $(OBJS)

