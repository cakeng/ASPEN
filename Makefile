TARGET=main
OBJECTS=util.o mat_mul.o cublas_mat_mul.o main.o cuda_aspen_tests.o

CXX=g++
CXXFLAGS= -Wall -fopenmp -O3 
# CXXFLAGS= -Wall -fopenmp -O0 -g

LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lopenblas -lgomp
COMMON=-I/usr/local/cuda/include/

ARCH= -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_61,code=[sm_61,compute_61] \
	  -gencode arch=compute_75,code=[sm_75,compute_75] \
	  -gencode arch=compute_80,code=[sm_80,compute_80] \
      -gencode arch=compute_86,code=[sm_86,compute_86] \

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(COMMON) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) 

%.o: %.cpp
	$(CXX) $(COMMON) $(CXXFLAGS) -c $< -o $@

mat_mul.o: mat_mul.cu
	nvcc $(ARCH) -c -o $@ $^

cublas_mat_mul.o: cublas_mat_mul.cu
	nvcc $(ARCH) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

