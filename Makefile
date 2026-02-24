# Makefile for ISP Pipeline Project

# 编译器设置
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fopenmp

# OpenCV 配置
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# OpenMP 配置
OPENMP_CFLAGS = -fopenmp
OPENMP_LIBS = -fopenmp

# 目录设置
SRC_DIR = src
CORE_DIR = $(SRC_DIR)/core
TESTS_DIR = $(SRC_DIR)/tests
BUILD_DIR = build
BM3D_DIR = bm3d

# 源文件
CORE_SOURCES = $(CORE_DIR)/raw_reader.cpp \
               $(CORE_DIR)/blc.cpp \
               $(CORE_DIR)/denoise.cpp \
               $(CORE_DIR)/awb.cpp \
               $(CORE_DIR)/gamma.cpp \
               $(CORE_DIR)/demosiac.cpp \
               $(CORE_DIR)/vng.cpp \
               $(CORE_DIR)/ahd.cpp \
               $(CORE_DIR)/AMaZE.cpp \
               $(CORE_DIR)/amazevng.cpp \
               $(CORE_DIR)/amazefromgithub.cpp \
               $(CORE_DIR)/rcd.cpp \
               $(CORE_DIR)/lmmse.cpp \
               $(CORE_DIR)/ccm.cpp \
               $(CORE_DIR)/sharpen.cpp \
               $(CORE_DIR)/bmp/converter.cpp \
               $(CORE_DIR)/bmp/nlm.cpp

TEST_SOURCES = $(TESTS_DIR)/test_gamma.cpp \
               $(TESTS_DIR)/test_vng.cpp \
               $(TESTS_DIR)/test_vng_opencv.cpp \
               $(TESTS_DIR)/test_ahd.cpp \
               $(TESTS_DIR)/test_amaze.cpp \
               $(TESTS_DIR)/test_amazevng.cpp \
               $(TESTS_DIR)/test_amazefromgithub.cpp \
               $(TESTS_DIR)/test_rcd.cpp \
               $(TESTS_DIR)/test_lmmse.cpp \
               $(TESTS_DIR)/test_rgb_ycbcr.cpp \
               $(TESTS_DIR)/test_rgb_ycbcr.cpp \
               $(TESTS_DIR)/test_nlm.cpp \
               $(TESTS_DIR)/test_no_nlm.cpp \
               $(TESTS_DIR)/test_bm3d.cpp

BM3D_SRC = $(BM3D_DIR)/src/bm3d_denoiser.cpp

# 目标文件
CORE_OBJECTS = $(BUILD_DIR)/raw_reader.o \
               $(BUILD_DIR)/blc.o \
               $(BUILD_DIR)/denoise.o \
               $(BUILD_DIR)/awb.o \
               $(BUILD_DIR)/gamma.o \
               $(BUILD_DIR)/demosiac.o \
               $(BUILD_DIR)/vng.o \
               $(BUILD_DIR)/ahd.o \
               $(BUILD_DIR)/AMaZE.o \
               $(BUILD_DIR)/amazevng.o \
               $(BUILD_DIR)/amazefromgithub.o \
               $(BUILD_DIR)/rcd.o \
               $(BUILD_DIR)/lmmse.o \
               $(BUILD_DIR)/ccm.o \
               $(BUILD_DIR)/sharpen.o \
               $(BUILD_DIR)/converter.o \
               $(BUILD_DIR)/nlm.o \
               $(BUILD_DIR)/dct.o \
               $(BUILD_DIR)/dwt.o

BM3D_OBJ = $(BUILD_DIR)/bm3d_denoiser.o

TEST_GAMMA_TARGET = $(BUILD_DIR)/test_gamma
TEST_VNG_TARGET = $(BUILD_DIR)/test_vng
TEST_VNG_OPENCV_TARGET = $(BUILD_DIR)/test_vng_opencv
TEST_AHD_TARGET = $(BUILD_DIR)/test_ahd
TEST_AMAZE_TARGET = $(BUILD_DIR)/test_amaze
TEST_AMAZEVNG_TARGET = $(BUILD_DIR)/test_amazevng
TEST_AMAZEFROMGITHUB_TARGET = $(BUILD_DIR)/test_amazefromgithub
TEST_RCD_TARGET = $(BUILD_DIR)/test_rcd
TEST_LMMSE_TARGET = $(BUILD_DIR)/test_lmmse
TEST_RGB_YCBCR_TARGET = $(BUILD_DIR)/test_rgb_ycbcr
TEST_NLM_TARGET = $(BUILD_DIR)/test_nlm
TEST_NO_NLM_TARGET = $(BUILD_DIR)/test_no_nlm
TEST_BM3D_TARGET = $(BUILD_DIR)/test_bm3d
TEST_MANUAL_DENOISE_TARGET = $(BUILD_DIR)/test_manual_denoise

# 默认目标
all: $(TEST_GAMMA_TARGET) $(TEST_VNG_TARGET) $(TEST_VNG_OPENCV_TARGET) $(TEST_AHD_TARGET) $(TEST_AMAZE_TARGET) $(TEST_AMAZEVNG_TARGET) $(TEST_AMAZEFROMGITHUB_TARGET) $(TEST_RCD_TARGET) $(TEST_LMMSE_TARGET) $(TEST_RGB_YCBCR_TARGET) $(TEST_NLM_TARGET) $(TEST_NO_NLM_TARGET) $(TEST_BM3D_TARGET) $(TEST_MANUAL_DENOISE_TARGET)

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

$(BUILD_DIR)/AMaZE.o: $(CORE_DIR)/AMaZE.cpp $(CORE_DIR)/AMaZE.h $(CORE_DIR)/demosiac.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/amazevng.o: $(CORE_DIR)/amazevng.cpp $(CORE_DIR)/amazevng.h $(CORE_DIR)/demosiac.h $(CORE_DIR)/vng.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/amazefromgithub.o: $(CORE_DIR)/amazefromgithub.cpp $(CORE_DIR)/amazefromgithub.h $(CORE_DIR)/demosiac.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/rcd.o: $(CORE_DIR)/rcd.cpp $(CORE_DIR)/rcd.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/lmmse.o: $(CORE_DIR)/lmmse.cpp $(CORE_DIR)/lmmse.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -c $< -o $@

$(BUILD_DIR)/ccm.o: $(CORE_DIR)/ccm.cpp $(CORE_DIR)/ccm.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/sharpen.o: $(CORE_DIR)/sharpen.cpp $(CORE_DIR)/sharpen.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/converter.o: $(CORE_DIR)/bmp/converter.cpp $(CORE_DIR)/bmp/converter.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/nlm.o: $(CORE_DIR)/bmp/nlm.cpp $(CORE_DIR)/bmp/nlm.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -c $< -o $@

$(BUILD_DIR)/dct.o: $(CORE_DIR)/bmp/dct.cpp $(CORE_DIR)/bmp/dct.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/dwt.o: $(CORE_DIR)/bmp/dwt.cpp $(CORE_DIR)/bmp/dwt.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 编译 BM3D 模块
$(BM3D_OBJ): $(BM3D_SRC) $(BM3D_DIR)/include/bm3d_denoiser.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -I$(BM3D_DIR)/include -c $< -o $@

# 编译测试程序
$(TEST_GAMMA_TARGET): $(TESTS_DIR)/test_gamma.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_VNG_TARGET): $(TESTS_DIR)/test_vng.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_VNG_OPENCV_TARGET): $(TESTS_DIR)/test_vng_opencv.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_AHD_TARGET): $(TESTS_DIR)/test_ahd.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_AMAZE_TARGET): $(TESTS_DIR)/test_amaze.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_AMAZEVNG_TARGET): $(TESTS_DIR)/test_amazevng.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_AMAZEFROMGITHUB_TARGET): $(TESTS_DIR)/test_amazefromgithub.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_RCD_TARGET): $(TESTS_DIR)/test_rcd.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS)

$(TEST_LMMSE_TARGET): $(TESTS_DIR)/test_lmmse.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS) $(OPENMP_LIBS)

$(TEST_RGB_YCBCR_TARGET): $(TESTS_DIR)/test_rgb_ycbcr.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS) $(OPENMP_LIBS)

$(TEST_NLM_TARGET): $(TESTS_DIR)/test_nlm.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS) $(OPENMP_LIBS)

$(TEST_NO_NLM_TARGET): $(TESTS_DIR)/test_no_nlm.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS) $(OPENMP_LIBS)

$(TEST_BM3D_TARGET): $(TESTS_DIR)/test_bm3d.cpp $(CORE_OBJECTS) $(BM3D_OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -I$(BM3D_DIR)/include -o $@ $< $(CORE_OBJECTS) $(BM3D_OBJ) $(OPENCV_LIBS) $(OPENMP_LIBS)

$(TEST_MANUAL_DENOISE_TARGET): $(TESTS_DIR)/test_manual_denoise.cpp $(CORE_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENMP_CFLAGS) -o $@ $< $(CORE_OBJECTS) $(OPENCV_LIBS) $(OPENMP_LIBS)

# 运行测试
run-gamma: $(TEST_GAMMA_TARGET)
	./$(TEST_GAMMA_TARGET)

run-vng: $(TEST_VNG_TARGET)
	./$(TEST_VNG_TARGET)

run-vng-opencv: $(TEST_VNG_OPENCV_TARGET)
	./$(TEST_VNG_OPENCV_TARGET)

run-ahd: $(TEST_AHD_TARGET)
	./$(TEST_AHD_TARGET)

run-amaze: $(TEST_AMAZE_TARGET)
	./$(TEST_AMAZE_TARGET)

run-amazevng: $(TEST_AMAZEVNG_TARGET)
	./$(TEST_AMAZEVNG_TARGET)

run-amazefromgithub: $(TEST_AMAZEFROMGITHUB_TARGET)
	./$(TEST_AMAZEFROMGITHUB_TARGET)

run-rcd: $(TEST_RCD_TARGET)
	./$(TEST_RCD_TARGET)

run-lmmse: $(TEST_LMMSE_TARGET)
	./$(TEST_LMMSE_TARGET)

run-rgb-ycbcr: $(TEST_RGB_YCBCR_TARGET)
	./$(TEST_RGB_YCBCR_TARGET)

run-nlm: $(TEST_NLM_TARGET)
	./$(TEST_NLM_TARGET)

run-no-nlm: $(TEST_NO_NLM_TARGET)
	./$(TEST_NO_NLM_TARGET)

run-bm3d: $(TEST_BM3D_TARGET)
	./$(TEST_BM3D_TARGET)

run-manual-denoise: $(TEST_MANUAL_DENOISE_TARGET)
	./$(TEST_MANUAL_DENOISE_TARGET)

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make                 - Build all tests"
	@echo "  make run-gamma       - Build and run test_gamma"
	@echo "  make run-vng         - Build and run test_vng"
	@echo "  make run-vng-opencv  - Build and run test_vng_opencv"
	@echo "  make run-ahd         - Build and run test_ahd"
	@echo "  make run-amaze       - Build and run test_amaze"
	@echo "  make run-amazevng    - Build and run test_amazevng (AMaZE-VNG Hybrid)"
	@echo "  make run-amazefromgithub - Build and run test_amazefromgithub (AMaZE From GitHub)"
	@echo "  make run-rcd         - Build and run test_rcd (RCD Demosaic)"
	@echo "  make run-lmmse       - Build and run test_lmmse (LMMSE Demosaic)"
	@echo "  make run-rgb-ycbcr   - Build and run test_rgb_ycbcr (RGB to YCbCr converter test)"
	@echo "  make run-nlm         - Build and run test_nlm (NLM Denoising)"
	@echo "  make run-no-nlm      - Build and run test_no_nlm (No NLM Denoising Comparison)"
	@echo "  make run-bm3d        - Build and run test_bm3d (BM3D Denoising)"
	@echo "  make run-manual-denoise - Build and run test_manual_denoise (DCT/DWT Manual Denoising)"
	@echo "  make clean           - Remove build files"
	@echo "  make help            - Show this help message"

.PHONY: all run-gamma run-vng run-vng-opencv run-ahd run-amaze run-amazevng run-amazefromgithub run-rcd run-lmmse run-rgb-ycbcr run-nlm run-bm3d run-manual-denoise clean help
