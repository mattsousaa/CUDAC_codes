all: folder app

NVCC = nvcc
BIN = bin
OBJ = obj
INC = -Iinc/
SRC_DIR = /home/matt/Desktop/CUDA_codes/Sum_Matrix/src
BIN_DIR = /home/matt/Desktop/CUDA_codes/Sum_Matrix/bin

app: main functions kernel_files functions
	@echo "Linking..."
	+@$(NVCC) -o $(BIN)/app $(OBJ)/main.o $(OBJ)/kernel_files.o $(OBJ)/functions.o
	@echo "Binary and text files are ready in $(BIN)/app"
	
main:
	+@echo "Compile: $@"
	@$(NVCC) src/$@.cu $(INC) -c -o $(OBJ)/$@.o

kernel_files:
	+@echo "Compile: $@"
	@$(NVCC) src/$@.cu $(INC) -c -o $(OBJ)/$@.o

functions:
	+@echo "Compile: $@"
	@$(NVCC) src/$@.cu $(INC) -c -o $(OBJ)/$@.o

folder:
	@mkdir -p $(OBJ) $(BIN)  
	
clean:
	@rm -rf $(OBJ)/* $(BIN)/* $(OBJ) $(BIN)