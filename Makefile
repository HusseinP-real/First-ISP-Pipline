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
               $(CORE_DIR)/denoise.cpp \
               $(CORE_DIR)/awb.cpp \
               $(CORE_DIR)/gamma.cpp \
               $(CORE_DIR)/demosiac.cpp \
               $(CORE_DIR)/vng.cpp \
               $(CORE_DIR)/ahd.cpp \
               $(CORE_DIR)/ccm.cpp \
               $(CORE_DIR)/sharpen.cpp
TEST_SOURCES = $(TESTS_DIR)/test_gamma.cpp \
               $(TESTS_DIR)/test_vng.cpp \
               $(TESTS_DIR)/test_vng_opencv.cpp \
               $(TESTS_DIR)/test_ahd.cpp

# 目标文件
CORE_OBJECTS = $(BUILD_DIR)/raw_reader.o \
               $(BUILD_DIR)/blc.o \
               $(BUILD_DIR)/denoise.o \
               $(BUILD_DIR)/awb.o \
               $(BUILD_DIR)/gamma.o \
               $(BUILD_DIR)/demosiac.o \
               $(BUILD_DIR)/vng.o \
               $(BUILD_DIR)/ahd.o \
               $(BUILD_DIR)/ccm.o \
               $(BUILD_DIR)/sharpen.o
TEST_GAMMA_TARGET = $(BUILD_DIR)/test_gamma
TEST_VNG_TARGET = $(BUILD_DIR)/test_vng
TEST_VNG_OPENCV_TARGET = $(BUILD_DIR)/test_vng_opencv
TEST_AHD_TARGET = $(BUILD_DIR)/test_ahd

# 默认目标
all: $(TEST_GAMMA_TARGET) $(TEST_VNG_TARGET) $(TEST_VNG_OPENCV_TARGET) $(TEST_AHD_TARGET)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译核心模块
$(BUILD_DIR)/raw_reader.o: $(CORE_DIR)/raw_reader.cpp $(CORE_DIR)/raw_reader.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/blc.o: $(CORE_DIR)/blc.cpp $(CORE_DIR)/blc.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/denoise.o: $(CORE_DIR)/denoise.cpp $(CORE_DIR)/denoise.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/awb.o: $(CORE_DIR)/awb.cpp $(CORE_DIR)/awb.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/gamma.o: $(CORE_DIR)/gamma.cpp $(CORE_DIR)/gamma.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/demosiac.o: $(CORE_DIR)/demosiac.cpp $(CORE_DIR)/demosiac.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/vng.o: $(CORE_DIR)/vng.cpp $(CORE_DIR)/vng.h $(CORE_DIR)/demosiac.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/ahd.o: $(CORE_DIR)/ahd.cpp $(CORE_DIR)/ahd.h $(CORE_DIR)/demosiac.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/ccm.o: $(CORE_DIR)/ccm.cpp $(CORE_DIR)/ccm.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/sharpen.o: $(CORE_DIR)/sharpen.cpp $(CORE_DIR)/sharpen.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# 编译测试程序
$(TEST_GAMMA_TARGET): $(TESTS_DIR)/test_gamma.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_VNG_TARGET): $(TESTS_DIR)/test_vng.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_VNG_OPENCV_TARGET): $(TESTS_DIR)/test_vng_opencv.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_AHD_TARGET): $(TESTS_DIR)/test_ahd.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

# 运行测试
run-gamma: $(TEST_GAMMA_TARGET)
	./$(TEST_GAMMA_TARGET)

run-vng: $(TEST_VNG_TARGET)
	./$(TEST_VNG_TARGET)

run-vng-opencv: $(TEST_VNG_OPENCV_TARGET)
	./$(TEST_VNG_OPENCV_TARGET)

run-ahd: $(TEST_AHD_TARGET)
	./$(TEST_AHD_TARGET)

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make              - Build test_gamma + test_vng + test_vng_opencv + test_ahd"
	@echo "  make run-gamma       - Build and run test_gamma"
	@echo "  make run-vng         - Build and run test_vng"
	@echo "  make run-vng-opencv  - Build and run test_vng_opencv"
	@echo "  make run-ahd         - Build and run test_ahd"
	@echo "  make clean           - Remove build files"
	@echo "  make help            - Show this help message"

.PHONY: all run-gamma run-vng run-vng-opencv run-ahd clean help

