TARGET=aspen_benchmark
GPU=1

CC=gcc
NVCC=nvcc

OPTS=-Ofast -march=native -funroll-loops
LDFLAGS= -L. -laspen -lm -lgomp 
COMMON= -I. 
CFLAGS= -fopenmp

ifeq ($(GPU), 1)
LDFLAGS+=-L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas
COMMON+=-I/usr/local/cuda/include/
endif

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(COMMON) $(CFLAGS) $(OPTS) $^ -o $@ $(LDFLAGS) $(ALIB)

clean:
	rm -rf $(TARGET)

