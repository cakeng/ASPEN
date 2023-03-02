TARGET=main
ALIB=libasapen.a
OBJECTS=apu.o

CC=gcc
NVCC=nvcc

OBJDIR=./obj/
VPATH=./src 
DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

# CFLAGS= -Wall -fopenmp -O3 -DDEBUG
CFLAGS= -Wall -fopenmp -O0 -g -DDEBUG
ARFLAGS=rcs
LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lopenblas -lgomp
COMMON=-I/usr/local/cuda/include/ -Iinclude/

ARCH= 	-gencode arch=compute_80,code=[sm_80,compute_80] \
		-gencode arch=compute_86,code=[sm_86,compute_86]
OBJS= $(addprefix $(OBJDIR), $(OBJECTS))
TOBJS= $(addsuffix .o, $(TARGET))

all: obj $(TARGET)

$(TARGET): $(TOBJS) $(ALIB) 
	$(CC) $(COMMON) $(CFLAGS) $< -o $@ $(TARGET) $(OBJS) $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

clean:
	rm -rf $(TARGET) $(OBJS)

