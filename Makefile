# Makefile for ISP Pipeline Project

# 编译器设置
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# OpenCV 配置
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# 目录设置
SRC_DIR = src
CORE_DIR = $(SRC_DIR)/core
TESTS_DIR = $(SRC_DIR)/tests
BUILD_DIR = build

# 源文件
CORE_SOURCES = $(CORE_DIR)/raw_reader.cpp \
               $(CORE_DIR)/blc.cpp \
               $(CORE_DIR)/awb.cpp \
               $(CORE_DIR)/gamma.cpp \
               $(CORE_DIR)/demosiac.cpp
TEST_SOURCES = $(TESTS_DIR)/test_raw_reader.cpp $(TESTS_DIR)/test_blc.cpp $(TESTS_DIR)/test_awb.cpp $(TESTS_DIR)/test_gamma.cpp $(TESTS_DIR)/trydemosiac.cpp

# 目标文件
CORE_OBJECTS = $(BUILD_DIR)/raw_reader.o \
               $(BUILD_DIR)/blc.o \
               $(BUILD_DIR)/awb.o \
               $(BUILD_DIR)/gamma.o \
               $(BUILD_DIR)/demosiac.o
TEST_TARGET = $(BUILD_DIR)/test_raw_reader
TEST_BLC_TARGET = $(BUILD_DIR)/test_blc
TEST_AWB_TARGET = $(BUILD_DIR)/test_awb
TEST_GAMMA_TARGET = $(BUILD_DIR)/test_gamma
TRY_DEMOSAIC_TARGET = $(BUILD_DIR)/trydemosiac

# 默认目标
all: $(TEST_TARGET) $(TEST_BLC_TARGET) $(TEST_AWB_TARGET) $(TEST_GAMMA_TARGET) $(TRY_DEMOSAIC_TARGET)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译核心模块
$(BUILD_DIR)/raw_reader.o: $(CORE_DIR)/raw_reader.cpp $(CORE_DIR)/raw_reader.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/blc.o: $(CORE_DIR)/blc.cpp $(CORE_DIR)/blc.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/awb.o: $(CORE_DIR)/awb.cpp $(CORE_DIR)/awb.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/gamma.o: $(CORE_DIR)/gamma.cpp $(CORE_DIR)/gamma.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/demosiac.o: $(CORE_DIR)/demosiac.cpp $(CORE_DIR)/demosiac.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# 编译测试程序
$(TEST_TARGET): $(TESTS_DIR)/test_raw_reader.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_BLC_TARGET): $(TESTS_DIR)/test_blc.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_AWB_TARGET): $(TESTS_DIR)/test_awb.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_GAMMA_TARGET): $(TESTS_DIR)/test_gamma.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TRY_DEMOSAIC_TARGET): $(TESTS_DIR)/trydemosiac.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

# 运行测试
run: $(TEST_TARGET)
	./$(TEST_TARGET)

run-blc: $(TEST_BLC_TARGET)
	./$(TEST_BLC_TARGET)

run-awb: $(TEST_AWB_TARGET)
	./$(TEST_AWB_TARGET)

run-gamma: $(TEST_GAMMA_TARGET)
	./$(TEST_GAMMA_TARGET)

run-demosiac: $(TRY_DEMOSAIC_TARGET)
	./$(TRY_DEMOSAIC_TARGET)

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make          - Build all test programs"
	@echo "  make run         - Build and run test_raw_reader"
	@echo "  make run-blc     - Build and run test_blc"
	@echo "  make run-awb     - Build and run test_awb"
	@echo "  make run-demosiac - Build and run trydemosiac"
	@echo "  make clean       - Remove build files"
	@echo "  make help        - Show this help message"

.PHONY: all run run-blc run-awb run-demosiac clean help

