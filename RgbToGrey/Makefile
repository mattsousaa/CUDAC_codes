all: folder app

NVCC = nvcc
CPP = g++
BIN = bin
OBJ = obj
INC = -Iinc/
SRC_DIR = /home/matt/Desktop/CUDA_codes/RgbToGrey/src
BIN_DIR = /home/matt/Desktop/CUDA_codes/RgbToGrey/bin

CUDA_INCLUDEPATH=/usr/local/cuda-10.2/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
OPENCV_PKGC=`pkg-config --cflags --libs opencv`

app: main student_func compare reference_calc
	@echo "Linking..."
	+@$(NVCC) -o $(BIN)/app $(OBJ)/main.o $(OBJ)/student_func.o $(OBJ)/compare.o $(OBJ)/reference_calc.o -I $(CUDA_INCLUDEPATH) -I $(OPENCV_LIBS) -I $(OPENCV_PKGC)
	@echo "Binary and text files are ready in $(BIN)/app"
	
main:
	+@echo "Compile: $@"
	@$(CPP) src/$@.cpp $(INC) -c -o $(OBJ)/$@.o -I $(CUDA_INCLUDEPATH) -I $(OPENCV_LIBS) -I $(OPENCV_PKGC)

student_func:
	+@echo "Compile: $@"
	@$(NVCC) src/$@.cu $(INC) -c -o $(OBJ)/$@.o -I $(CUDA_INCLUDEPATH) -I $(OPENCV_LIBS) -I $(OPENCV_PKGC)

compare:
	+@echo "Compile: $@"
	@$(CPP) src/$@.cpp $(INC) -c -o $(OBJ)/$@.o -I $(CUDA_INCLUDEPATH) -I $(OPENCV_LIBS) -I $(OPENCV_PKGC)

reference_calc:
	+@echo "Compile: $@"
	@$(CPP) src/$@.cpp $(INC) -c -o $(OBJ)/$@.o -I $(CUDA_INCLUDEPATH) -I $(OPENCV_LIBS) -I $(OPENCV_PKGC)

folder:
	@mkdir -p $(OBJ) $(BIN)
	@cp $(SRC_DIR)/*.jpg $(BIN_DIR)  
	
clean:
	@rm -rf $(OBJ)/* $(BIN)/* $(OBJ) $(BIN)