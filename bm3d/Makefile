# Makefile for BM3D Denoiser Project

# 编译器设置
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native
CXXFLAGS_DEBUG = -std=c++17 -Wall -Wextra -g -O0

# OpenCV 配置
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# OpenMP 配置（可选，用于并行化）
OPENMP_CFLAGS = -fopenmp
OPENMP_LIBS = -fopenmp

# 目录设置
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# 源文件
BM3D_SRC = $(SRC_DIR)/bm3d_denoiser.cpp
CONVERTER_SRC = ../src/core/bmp/converter.cpp
MAIN_SRC = main.cpp

# 目标文件
BM3D_OBJ = $(BUILD_DIR)/bm3d_denoiser.o
CONVERTER_OBJ = $(BUILD_DIR)/converter.o
BM3D_LIB = $(BUILD_DIR)/libbm3d.a
TARGET = $(BUILD_DIR)/bm3d_denoise

# 默认目标
all: $(TARGET)

# Debug 构建
debug: CXXFLAGS = $(CXXFLAGS_DEBUG)
debug: $(TARGET)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译 BM3D 库对象文件
$(BM3D_OBJ): $(BM3D_SRC) $(INCLUDE_DIR)/bm3d_denoiser.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# 编译 Converter 对象文件
$(CONVERTER_OBJ): $(CONVERTER_SRC) ../src/core/bmp/converter.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# 创建静态库
$(BM3D_LIB): $(BM3D_OBJ)
	ar rcs $@ $<

# 编译主程序
$(TARGET): $(MAIN_SRC) $(BM3D_LIB) $(CONVERTER_OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -I$(INCLUDE_DIR) -I../src/core/bmp -o $@ $< $(BM3D_LIB) $(CONVERTER_OBJ) $(OPENCV_LIBS) $(OPENMP_LIBS)

# 运行程序（使用默认参数）
run: $(TARGET)
	@echo "Running BM3D denoising with default parameters..."
	@echo "Input: ../data/input, Sigma: 25, Output: ../data/output"
	./$(TARGET) ../data/input 25 ../data/output

# 运行程序（自定义参数）
# 使用方法: 
#   make run-custom SIGMA=30
#   make run-custom SIGMA=30 INPUT=../data/input
#   make run-custom SIGMA=30 INPUT=../data/input OUTPUT=../data/output
run-custom: $(TARGET)
	@if [ -z "$(SIGMA)" ]; then \
		echo "Error: SIGMA is required."; \
		echo "Usage: make run-custom SIGMA=25 [INPUT=../data/input] [OUTPUT=../data/output]"; \
		exit 1; \
	fi
	@ARGS="$(SIGMA)"; \
	if [ -n "$(INPUT)" ]; then ARGS="$$ARGS $(INPUT)"; fi; \
	if [ -n "$(OUTPUT)" ]; then ARGS="$$ARGS $(OUTPUT)"; fi; \
	./$(TARGET) $$ARGS

# 快速测试（处理单个文件）
# 使用方法: make test SIGMA=25 INPUT_FILE=../data/input/test.bmp
test: $(TARGET)
	@if [ -z "$(SIGMA)" ] || [ -z "$(INPUT_FILE)" ]; then \
		echo "Error: SIGMA and INPUT_FILE are required."; \
		echo "Usage: make test SIGMA=25 INPUT_FILE=../data/input/test.bmp"; \
		exit 1; \
	fi
	@mkdir -p $(BUILD_DIR)/test_output
	./$(TARGET) $(SIGMA) $(dir $(INPUT_FILE)) $(BUILD_DIR)/test_output

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 完全清理（包括输出文件）
distclean: clean
	@echo "Note: This does not remove output images. Remove them manually if needed."

# 显示帮助信息
help:
	@echo "BM3D Denoiser Makefile"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build the bm3d_denoise executable (Release mode)"
	@echo "  make debug        - Build in debug mode"
	@echo "  make run          - Build and run with default parameters"
	@echo "  make run-custom   - Build and run with custom parameters"
	@echo "                     Usage: make run-custom SIGMA=25 [INPUT=../data/input] [OUTPUT=../data/output]"
	@echo "  make test         - Build and test with a single file"
	@echo "                     Usage: make test SIGMA=25 INPUT_FILE=../data/input/test.bmp"
	@echo "  make clean        - Remove build files"
	@echo "  make distclean    - Remove all build files"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                                    # Build"
	@echo "  make run                                # Build and run with defaults"
	@echo "  make run-custom SIGMA=30                # Run with sigma=30 (uses defaults for input/output)"
	@echo "  make run-custom SIGMA=25 INPUT=../data/input  # Run with custom input"
	@echo "  make run-custom SIGMA=25 INPUT=../data/input OUTPUT=../data/output"
	@echo "  make test SIGMA=25 INPUT_FILE=../data/input/test.bmp"

.PHONY: all debug run run-custom test clean distclean help
