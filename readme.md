<div align="center">
<h1>🔥 消防隐患识别智慧骑士系统 (WiseKnight) 🔥</h1>
</div>

<div align="center">
<img src="assets/images/logo.png" alt="WiseKnightSystem" width="150" height="150">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
</div>

## 🌟 系统概述

**消防隐患识别智慧骑士系统**是一个基于视觉语言模型的智能分析平台，专门用于识别和分析楼道场景中的消防隐患。系统能够自动检测楼道环境，评估消防隐患等级。

### 核心功能
- 🏢 **楼道场景识别**：自动识别图像是否为楼道场景
- ⚠️ **风险等级评估**：智能分析楼道内的消防隐患，划分风险等级
- 🧠 **深度思考分析**：可以选择启用深度思考进行分析并展示大模型的思考过程
- 📊 **可视化报告**：美观的可视化界面展示分析结果
- 🔄 **实时交互**：支持流式输出和实时进度显示

## 🏗️ 系统架构

### 核心技术栈
- **后端框架**：Gradio + FastAPI
- **模型框架**：Unsloth + Transformers
- **视觉模型**：基于LLaMA-3的视觉语言模型
- **前端界面**：HTML + CSS + JavaScript

### 系统特点
1. **模块化设计**：各功能模块独立，便于扩展和维护
2. **流式输出**：支持实时显示思考进度
3. **离线支持**：可完全离线运行
4. **响应式界面**：适配不同设备和屏幕尺寸

## 🚀 快速开始

<details style="color:rgb(128,128,128)">
<summary>分享本人的软硬件配置（仅供参考）</summary>

* CPU: 12th Gen Intel(R) Core(TM) i7-12700F (20核心)
* RAM: 32 GB
* GPU: NVIDIA GeForce RTX 4070 (12GB) * 1
* Ubuntu==22.04
* CUDA==12.4
* Python==3.10.19
* [requirements.txt](./requirements.txt)
</details>

### 安装步骤

1. **克隆代码库**
```bash
git clone https://github.com/Zhen-x-l/WiseKnight.git
```

2. **进入WiseKnight项目文件夹**
```bash
cd WiseKnight
```

3. **创建并激活一个用作该项目的conda环境**
```bash
conda create -n WiseKnight python=3.10
conda activate WiseKnight
```

4. **在环境中安装依赖包**
```bash
pip install -r requirements.txt
```

5. **启动Web应用**
```bash
python web_WiseKnight.py
```

### 参数说明
```bash
python web_WiseKnight.py --help

--model_path     模型路径，默认model/llama-3-2-11b-vision-instruct-4bit-r16-think/last_v2
--device         运行设备，默认cuda，可设置为cpu
--max_seq_len    最大序列长度，默认4096
--port           服务器端口，默认8888
--share          创建公开可访问的链接，默认False
--logo_path      自定义logo图片路径，默认assets/images/logo.png
```

## 🖥️ 使用指南

### 1. 上传图片
- 点击图片上传区域或拖拽图片到指定区域即可上传要分析的图片
- 支持JPG、PNG、JPEG格式，最大10MB

### 2. 配置分析选项
- **深度思考模式**：启用后显示模型的思考步骤
- **分析标准说明**：查看详细的风险评估规则

### 3. 开始分析
- 点击"🔍 开始分析"按钮即可开始分析
- 先点击"🧠 深度思考"按钮再点击"🔍 开始分析"按钮即可启用深度思考进行分析并展示大模型的思考过程

### 4. 查看结果
- **风险等级标签**：显示评估结果（高风险/中风险/低风险/无风险/非楼道）
- **分析过程**：如启用深度思考模式，显示详细思考步骤
- **可视化展示**：使用不同颜色和图标直观展示风险等级

## ⚠️ 风险评估标准

### 风险等级定义
| 风险等级 | 颜色 | 含义描述 | 说明 |
|---------|------|------|------|
| **高风险** | 🔴 红色 | 存在严重安全隐患 | 存在电动自行车、电池充电设备或飞线充电 |
| **中风险** | 🟠 橙色 | 存在一定安全隐患 | 有大量杂物，严重阻碍通行或包含明显可燃物品 |
| **低风险** | 🟢 绿色 | 基本安全 | 有少量物品或摆放整齐，仅轻微影响通行 |
| **无风险** | 🔵 蓝色 | 非常安全 | 走廊干净无存放物 |
| **非楼道** | ⚪ 灰色 | 非楼道场景 | 图像不属于楼道场景 |

## 🎯 使用示例

data_sample文件夹下提供了上述五类数据的样例数据，每类样例数据各5张可以用来展示使用。

<div align="center">
<h1><strong>消防隐患识别智慧骑士系统使用展示</strong><h1>

![WiseKnightSystem](assets/images/show.gif)
</div>