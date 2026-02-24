# ISP Pipeline Project

图像信号处理（Image Signal Processing）Pipeline项目 (C++)

## 项目结构

详细的项目结构说明请参考 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)

## 项目结构概览

```
First-ISP-Pipline/
├── src/              # 源代码
│   ├── core/         # 核心pipeline和I/O
│   ├── modules/      # ISP处理模块
│   ├── utils/        # 工具函数
│   └── tests/        # 单元测试
├── include/          # 头文件（可选）
├── configs/          # 配置文件
├── data/             # 数据文件
├── scripts/          # 脚本工具
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

## 参考文档

- [项目结构详细说明](./PROJECT_STRUCTURE.md)

