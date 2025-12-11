# ISP Pipeline Project

图像信号处理（Image Signal Processing）Pipeline项目

## 项目结构

详细的项目结构说明请参考 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)

## 快速开始

### 1. 创建项目目录结构

```bash
mkdir -p src/{core,modules,utils,tests}
mkdir -p configs/{camera_profiles,tuning_params}
mkdir -p data/{input,output,reference}
mkdir -p scripts docs
```

### 2. 安装依赖

```bash
pip install numpy opencv-python pillow pyyaml matplotlib pytest
```

### 3. 项目结构概览

```
pusenProject1/
├── src/              # 源代码
│   ├── core/         # 核心pipeline和I/O
│   ├── modules/      # ISP处理模块
│   └── utils/        # 工具函数
├── configs/          # 配置文件
├── data/             # 数据文件
├── scripts/          # 运行脚本
└── docs/             # 文档
```

## ISP处理流程

典型的ISP pipeline处理顺序：

1. RAW图像读取
2. 黑电平校正
3. 去马赛克（Demosaicing）
4. 白平衡
5. 色彩校正
6. 伽马校正
7. 降噪
8. 锐化
9. 输出处理后的图像

## 开发建议

1. 从核心模块开始：先实现 `raw_reader.py` 和 `pipeline.py`
2. 逐步添加处理模块：按处理顺序逐个实现
3. 使用配置文件管理参数，便于调优
4. 为每个模块编写单元测试

## 参考文档

- [项目结构详细说明](./PROJECT_STRUCTURE.md)

