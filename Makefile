# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O3
ifeq (,$(shell which nvprof))
    NVCC_FLAGS += -arch=sm_20
endif

# OpenCV libraries
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv)

# Target executable
EXE := histogram

# Source files
SRCS := main.cu support.cu

# Object files
OBJS := $(SRCS:.cu=.o)

# Rule to build the executable
$(EXE): $(OBJS)
	$(NVCC) $(OBJS) -o $(EXE) $(NVCC_FLAGS) $(OPENCV_FLAGS)

# Rule to compile CUDA source files
%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCC_FLAGS)

# Phony target to clean up the directory
.PHONY: clean
clean:
	rm -f $(OBJS) $(EXE)
