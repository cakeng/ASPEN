TARGET=main
OBJECTS=util.o main.o cuda_aspen_tests.o cuda_aspen.o aspen_pthread.o

CXX=g++
CXXFLAGS= -Wall -fopenmp -O3 -DDEBUG
# CXXFLAGS= -Wall -fopenmp -O0 -g -DDEBUG

LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lopenblas -lgomp
COMMON=-I/usr/local/cuda/include/

ARCH= 	-gencode arch=compute_80,code=[sm_80,compute_80] \
		-gencode arch=compute_86,code=[sm_86,compute_86] \

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(COMMON) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) 

%.o: %.cpp
	$(CXX) $(COMMON) $(CXXFLAGS) -c $< -o $@

cuda_aspen.o: cuda_aspen_tests.cu
	nvcc $(ARCH) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

