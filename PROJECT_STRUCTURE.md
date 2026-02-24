# ISP Pipeline 项目结构 (C++)

## 项目目录结构

```
First-ISP-Pipline/
├── src/                          # 源代码目录
│   ├── core/                     # 核心ISP处理实现
│   │   ├── bmp/                  # BMP处理与高级降噪 (NLM, BM3D, DCT, DWT)
│   │   ├── raw_reader.cpp/h      # RAW文件读取与解析
│   │   ├── demosiac.cpp/h        # 去马赛克基础接口
│   │   ├── ahd.cpp/h             # AHD 去马赛克实现
│   │   ├── amaze.cpp/h           # AMaZE 去马赛克实现
│   │   ├── rcd.cpp/h             # RCD 去马赛克实现
│   │   └── ...                   # 其他模块 (AWB, BLC, CCM, Gamma等)
│   ├── modules/                  # ISP 各个功能模块
│   ├── utils/                    # 工具函数
│   └── tests/                    # 单元测试与算法对比脚本
│       ├── test_nlm.cpp          # NLM 降噪测试
│       ├── test_bm3d.cpp         # BM3D 降噪测试
│       └── ...
├── include/                      # 头文件目录（可选）
├── configs/                      # 配置文件目录
├── data/                         # 数据目录
│   ├── input/                    # 输入源文件 (RAW/BMP)
│   ├── output/                   # 处理结果 (gitignored)
│   └── reference/                # 参考图像
├── scripts/                      # 辅助脚本
├── docs/                         # 文档目录
├── build/                        # 编译输出目录
├── Makefile                      # 核心构建系统
├── CMakeLists.txt                # 替代构建系统
├── README.md                     # 项目说明
└── .gitignore                    # 忽略项管理
```

## 目录说明

### src/
- **core/**: 核心算法实现
  - **bmp/**: 处理 BMP 格式后的进阶降噪算法，包含 NLM, BM3D, DCT, DWT 等从零实现的算法。
  - **demosaic/**: 实现多种去马赛克算法（AHD, RCD, AMaZE 等）。
  - **ISP Pipeline**: 包含黑电平校正 (BLC)、白平衡 (AWB)、色彩矩阵 (CCM)、伽马校正等基础模块。
- **modules/**: ISP各个处理模块（黑电平、去马赛克、白平衡等）
- **utils/**: 工具函数（图像处理、数学运算、配置加载等）
- **tests/**: 包含针对不同算法的独立测试程序，可生成对比图像。

### include/ (可选)
- 如果使用头文件和源文件分离的结构，可以将头文件放在这里

### configs/
- **camera_profiles/**: 不同相机/传感器的配置文件
- **tuning_params/**: ISP参数调优配置文件

### data/
- **input/**: 输入的RAW图像文件
- **output/**: 处理后的输出图像
- **reference/**: 参考图像（用于对比和测试）

### scripts/
- 辅助脚本（批量处理、可视化等）

### docs/
- 项目文档（架构说明、算法文档、API参考等）

## 设计原则

1. **模块化**: 每个ISP处理步骤独立成模块，便于测试和维护
2. **可配置**: 所有参数通过配置文件管理，避免硬编码
3. **可扩展**: 易于添加新的处理模块
4. **可测试**: 每个模块都有对应的单元测试
5. **性能优化**: C++原生性能，可考虑SIMD、多线程、GPU加速等

## 技术栈建议

- **C++**: 主要开发语言（建议C++17或更高版本）
- **CMake**: 构建系统（推荐）
- **OpenCV**: 图像处理库（可选）
- **yaml-cpp**: YAML配置文件解析（可选）
- **Google Test**: 单元测试框架（可选）

## ISP处理流程

典型的ISP pipeline处理顺序：
1. 黑电平校正
2. 去马赛克（Bayer → RGB）
3. 白平衡
4. 色彩校正（CCM）
5. 伽马校正
6. 降噪
7. 锐化
8. 色调映射（可选，用于HDR）

