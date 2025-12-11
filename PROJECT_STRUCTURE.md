# ISP Pipeline 项目结构建议

## 项目目录结构

```
pusenProject1/
├── src/                          # 源代码目录
│   ├── core/                     # 核心ISP处理模块
│   │   ├── __init__.py
│   │   ├── pipeline.py           # ISP主pipeline类
│   │   ├── raw_reader.py         # RAW图像读取器
│   │   └── image_writer.py       # 处理后图像写入器
│   │
│   ├── modules/                  # ISP各个处理模块
│   │   ├── __init__.py
│   │   ├── black_level.py        # 黑电平校正
│   │   ├── demosaic.py           # 去马赛克（Bayer转RGB）
│   │   ├── white_balance.py      # 白平衡
│   │   ├── color_correction.py   # 色彩校正（CCM）
│   │   ├── gamma.py              # 伽马校正
│   │   ├── denoise.py            # 降噪
│   │   ├── sharpen.py            # 锐化
│   │   └── tone_mapping.py       # 色调映射（HDR）
│   │
│   ├── utils/                    # 工具函数
│   │   ├── __init__.py
│   │   ├── image_utils.py        # 图像处理工具函数
│   │   ├── math_utils.py         # 数学工具函数
│   │   └── config_loader.py      # 配置文件加载器
│   │
│   ├── tests/                    # 单元测试
│   │   ├── __init__.py
│   │   ├── test_pipeline.py
│   │   ├── test_demosaic.py
│   │   └── test_white_balance.py
│   │
│   └── raw1.raw, raw2.raw, ...   # RAW图像文件（已存在）
│
├── configs/                      # 配置文件目录
│   ├── default.yaml              # 默认ISP参数配置
│   ├── camera_profiles/          # 不同相机/传感器的配置
│   │   ├── sony_imx586.yaml
│   │   └── custom.yaml
│   └── tuning_params/            # 调优参数
│       └── tuning_template.yaml
│
├── data/                         # 数据目录
│   ├── input/                    # 输入RAW文件（可移动src中的文件到这里）
│   ├── output/                   # 处理后的输出图像
│   └── reference/                # 参考图像（用于对比）
│
├── scripts/                      # 脚本工具
│   ├── run_pipeline.py           # 运行ISP pipeline的主脚本
│   ├── batch_process.py          # 批量处理脚本
│   └── visualize_results.py      # 结果可视化脚本
│
├── docs/                         # 文档目录
│   ├── architecture.md           # 架构说明
│   ├── algorithms.md             # 算法说明
│   └── api_reference.md          # API参考文档
│
├── requirements.txt              # Python依赖
├── setup.py                      # 安装脚本（可选）
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```

## 模块设计说明

### 1. core/pipeline.py
- **职责**: ISP pipeline的主控制器
- **功能**:
  - 管理各个处理模块的执行顺序
  - 处理模块间的数据传递
  - 错误处理和日志记录
  - 性能统计

### 2. core/raw_reader.py
- **职责**: 读取RAW图像文件
- **功能**:
  - 支持不同格式的RAW文件（.raw, .dng等）
  - 解析RAW文件头信息（分辨率、Bayer模式、位深等）
  - 将RAW数据转换为numpy数组

### 3. modules/ 各个处理模块
每个模块应该：
- 实现独立的处理算法
- 有清晰的输入/输出接口
- 支持参数配置
- 可单独测试

**典型ISP处理流程**:
1. **black_level.py**: 黑电平校正（减去暗电流）
2. **demosaic.py**: Bayer模式转RGB（去马赛克）
3. **white_balance.py**: 白平衡校正
4. **color_correction.py**: 色彩矩阵校正（CCM）
5. **gamma.py**: 伽马曲线校正
6. **denoise.py**: 降噪处理
7. **sharpen.py**: 锐化增强
8. **tone_mapping.py**: 色调映射（可选，用于HDR）

### 4. utils/ 工具模块
- **image_utils.py**: 图像操作工具（裁剪、缩放、格式转换等）
- **math_utils.py**: 数学运算工具（矩阵运算、插值等）
- **config_loader.py**: 加载YAML/JSON配置文件

### 5. configs/ 配置管理
- 使用YAML格式存储ISP参数
- 支持不同相机/传感器的配置
- 便于调优和实验

## 设计原则

1. **模块化**: 每个ISP处理步骤独立成模块，便于测试和维护
2. **可配置**: 所有参数通过配置文件管理，避免硬编码
3. **可扩展**: 易于添加新的处理模块
4. **可测试**: 每个模块都有对应的单元测试
5. **性能优化**: 考虑使用numpy、OpenCV等高效库，或C++扩展

## 技术栈建议

- **Python**: 主要开发语言
- **NumPy**: 数组操作和数值计算
- **OpenCV**: 图像处理（可选，用于某些算法）
- **Pillow/PIL**: 图像I/O
- **PyYAML**: 配置文件解析
- **Matplotlib**: 结果可视化
- **pytest**: 单元测试框架

## 可选的高级特性

1. **GPU加速**: 使用CuPy或PyTorch进行GPU加速
2. **C++扩展**: 关键算法用C++实现，通过pybind11绑定
3. **并行处理**: 多线程/多进程处理多张图像
4. **实时预览**: 添加实时处理预览功能
5. **参数调优工具**: 图形界面工具用于参数调优

## 下一步建议

1. 先实现核心的pipeline框架和RAW读取器
2. 逐个实现各个ISP模块（建议顺序：黑电平 → 去马赛克 → 白平衡 → 色彩校正 → 伽马）
3. 添加配置文件和参数管理
4. 编写单元测试
5. 优化性能和添加高级特性

