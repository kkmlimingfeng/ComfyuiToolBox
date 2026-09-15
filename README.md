# ComfyUI Toolbox

> 面向 **ComfyUI / Stable Diffusion / LoRA 工作流** 的本地工具箱。
>
> 用一个轻量的 Web 门户统一管理多个独立工具，让常用的图片、Prompt、LoRA 与数据集小工具可以快速启动、集中使用，同时保持彼此独立。

<!--
首屏展示图预留：
建议放一张 16:9 或 3:2 的门户总览截图。

![ComfyUI Toolbox](docs/images/portal.png)
-->

![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D6?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square)
![Local](https://img.shields.io/badge/Local--First-✓-2EA043?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Overview

**ComfyUI Toolbox** 不是一个“大而全”的单体应用，而是一套可以持续扩展的本地工具集合。

项目提供一个统一门户，负责发现、启动、停止和切换工具；具体功能仍由各自的独立工具负责。

```text
                      ComfyUI Toolbox
                             │
                    ┌────────┴────────┐
                    │   Local Portal  │
                    │  Tool Discovery │
                    │ Process Manager │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   Tag Editor          Image Prompt          LoRA Inspect
        │                    │                    │
        └────────────── 独立 Flask 服务 ───────────┘
                             │
                       更多工具持续加入
```

### 设计目标

- **一键进入**：统一 `run.bat` 启动门户。
- **工具独立**：每个工具拥有自己的目录、入口和端口，也可以单独运行。
- **低耦合扩展**：通过 `manifest.json` 自动发现工具，不需要反复修改门户页面。
- **本地优先**：图片、标签和模型文件默认在本机处理。
- **适合个人工作流**：优先解决 ComfyUI / SD / LoRA 使用过程中频繁出现的小问题。

---

## ✨ Features

| 工具 | 用途 | 状态 |
| --- | --- | :---: |
| 🏷️ 图片标签编辑器 | 浏览图片、编辑 `.txt` 标签、批量增删/排序、离线翻译 | ✅ |
| 🖼️ Krea2 提示词反推 | 使用本地视觉语言模型从图片生成文生图 Prompt | ✅ |
| 🔍 LoRA 参数分析 | 查看网络算法、Rank / Alpha、训练信息及作用层级 | ✅ |
| 📂 批量文件后缀更改 | 批量追加 / 删除文件名字符串，支持预览和防覆盖 | ✅ |
| 🛒 Danbooru 标签超市 | 查询、组合和导出 Danbooru 标签 | ✅ |

> **后续新增工具** 会继续沿用相同的门户接入方式，因此 README 的工具表也可以按同样格式继续向下扩展。

<!--
工具截图预留区域

### 图片标签编辑器
![图片标签编辑器](docs/images/editor.png)

### Krea2 提示词反推
![Krea2 提示词反推](docs/images/img2prompt.png)

### LoRA 参数分析
![LoRA 参数分析](docs/images/lora-inspect.png)

### 批量文件后缀更改
![批量文件后缀更改](docs/images/renamer.png)

### Danbooru 标签超市
![Danbooru 标签超市](docs/images/danbooru.png)
-->

---

## 🚀 Quick Start

### 1. 准备项目自带 Python

本项目的启动脚本**只使用项目目录内的 Python**，不会调用 Windows 系统中的 `python` 或 `py`。

确保项目结构中存在：

```text
ComfyUI-Toolbox/
└── python/
    └── python.exe
```

推荐直接使用项目 Release 提供的完整运行包。

> GitHub 源码仓库不建议直接提交整个 Python 运行时、模型和大型二进制文件。源码版使用前，应先将项目要求的 Python 运行时放入 `python\` 目录。

### 2. 安装依赖

部分工具需要 `llama-cpp-python`。

Windows / Python 3.12 / CUDA 12.8 可以使用：

`llama_cpp_python-0.3.49+cu128-cp312-cp312-win_amd64.whl`

下载页面，下载好需放在项目根目录：

<https://github.com/JamePeng/llama-cpp-python/releases>

首次运行：

```bat
setup_env.bat
```

脚本会使用：

```text
python\python.exe
```

安装项目依赖，并检查 PyTorch / CUDA 等核心组件。

#### 🤖 模型

项目中的部分功能使用 Hugging Face 上公开发布的模型。

**Hy-MT2 1.8B**

用于本地文本翻译。

模型仓库：

<https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF>

当前使用：

```text
Hy-MT2-1.8B-Q4_K_M.gguf
```

**Qwen3-VL-4B-FP8**

用于视觉语言相关功能。

模型仓库：

<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8>

### 3. 启动门户

```bat
run.bat
```

启动后打开门户页面，即可从侧边栏切换各个工具。

### 4. 单独运行某个工具

每个工具都可以脱离门户独立运行：

```text
tools/
├── editor/run.bat
tools/
├── img2prompt/run.bat
tools/
├── lora_inspect/run.bat
tools/
└── renamer/run.bat
```

例如：

```bat
cd tools\editor
run.bat
```

---

## 🧩 Tool Details

### 🏷️ 图片标签编辑器

用于图片数据集的标签整理与 Prompt 编辑。

支持：

- 图片与对应 `.txt` 标签浏览
- 标签添加、删除、排序
- 批量修改
- 本地标签翻译
- 与常见 Danbooru / SD 标签工作流配合

<!-- 截图预留：docs/images/editor-detail.png -->

### 🖼️ Krea2 提示词反推

使用本地视觉语言模型，根据输入图片生成适用于文生图模型的 Prompt。

适合用于：

- 图片提示词整理
- 数据集辅助标注
- 参考图 Prompt 提取
- 中英文 Prompt 生成

需要 NVIDIA GPU 及对应模型文件。

<!-- 截图预留：docs/images/img2prompt-detail.png -->

### 🔍 LoRA 参数分析

读取 LoRA / LyCORIS 等模型文件中的 metadata 与网络信息，帮助快速判断模型的训练配置。

可关注：

- Network / LoRA 算法
- Rank
- Alpha
- Base Model
- Epoch / Step
- Learning Rate
- 作用层级

<!-- 截图预留：docs/images/lora-detail.png -->

### 📂 批量文件后缀更改

面向数据集整理的轻量批处理工具。

提供预览、跳过与防覆盖机制，避免批量重命名时误操作。

<!-- 截图预留：docs/images/renamer-detail.png -->

### 🛒 Danbooru 标签超市

用于 Danbooru 标签查询、组合和导出。

该工具依赖第三方网络服务，具体可用性受网络环境和第三方服务状态影响。

<!-- 截图预留：docs/images/danbooru-detail.png -->

---

## 🧠 Models

模型文件默认不提交到 GitHub 仓库。

### Hy-MT2-1.8B

用于图片标签编辑器中的本地翻译。

建议放置：

```text
model/
└── Hy-MT2-1.8B-Q4_K_M.gguf
```

标签编辑器会在需要翻译时加载该模型。

模型来源：`tencent/Hy-MT2-1.8B-GGUF`

### Qwen3-VL-4B-Instruct-FP8

用于 Krea2 提示词反推。

建议放置：

```text
model/
└── Qwen3-VL-4B-Instruct-FP8/
    └── ...
```

视觉模型不会因为启动门户而自动加载，通常是在进入对应工具后按需加载，从而避免门户启动时立即占用大量显存。

---

## ⚙️ Optional: llama-cpp-python

图片标签编辑器的离线翻译功能依赖 `llama-cpp-python`。

为了避免 Windows 环境从源码编译 CUDA 依赖，项目支持使用预编译 wheel。

把匹配当前 Python / CUDA 环境的 wheel 放在项目根目录，例如：

```text
llama_cpp_python-*.whl
```

然后重新运行：

```bat
setup_env.bat
```

没有该 wheel 时，其他不依赖 `llama-cpp-python` 的工具仍可以使用；标签编辑器本身也可以启动，但离线翻译功能不可用。

---

## 📁 Project Structure

```text
ComfyUI-Toolbox/
│
├── python/                    # 项目运行时（Release / 本地部署准备）
│   └── python.exe
│
├── model/                     # 本地模型，不提交到 Git
├── docs/
│   └── images/                # README 展示截图
│
├── portal.py                  # 统一门户
├── run.bat                    # 启动门户
├── setup_env.bat              # 安装 / 检查环境
├── requirements.txt           # 通用 Python 依赖
├── LICENSE
├── README.md
├── .gitignore
│
└── tools/
    ├── editor/
    │   ├── app.py
    │   ├── manifest.json
    │   └── run.bat
    │
    ├── img2prompt/
    │   ├── app.py
    │   ├── manifest.json
    │   └── run.bat
    │
    ├── lora_inspect/
    │   ├── app.py
    │   ├── inspect_core.py
    │   ├── manifest.json
    │   └── run.bat
    │
    └── renamer/
        ├── app.py
        ├── manifest.json
        └── run.bat
```

---

## ➕ Add a New Tool

门户采用 **Manifest-driven** 的工具发现机制。

新增工具时，推荐保持以下结构：

```text
tools/
└── my_tool/
    ├── app.py
    ├── manifest.json
    └── run.bat
```

### `manifest.json`

```json
{
  "id": "my_tool",
  "group": "本地工具箱",
  "name": "我的工具",
  "icon": "🔧",
  "port": 7870,
  "script": "app.py",
  "desc": "一句话说明这个工具做什么"
}
```

| 字段 | 必填 | 说明 |
| --- | :---: | --- |
| `id` | ✅ | 全局唯一 ID |
| `group` | ✅ | 门户菜单分组 |
| `name` | ✅ | 门户显示名称 |
| `icon` | ❌ | 菜单图标 |
| `port` | ✅ | 工具监听端口 |
| `script` | ❌ | 启动脚本，默认 `app.py` |
| `desc` | ❌ | 工具简介 |

### `/ping` 健康检查

门户通过 `/ping` 判断工具是否已经正常运行。

最小 Flask 示例：

```python
from flask import Flask

app = Flask(__name__)


@app.get("/ping")
def ping():
    return {"ok": True}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7870)
```

### 接入流程

```text
创建 tools/<tool>/
        ↓
添加 manifest.json
        ↓
实现 app.py + /ping
        ↓
添加 run.bat
        ↓
重启 portal
        ↓
工具自动出现在侧边栏
```

新工具建议始终监听 `127.0.0.1`，除非确实需要向局域网开放服务。

---

## 🔌 Port Map

| 服务 | 端口 |
| --- | ---: |
| Portal | `7860` |
| Reserved / legacy | `7861`–`7863` |
| LoRA 参数分析 | `7864` |
| 批量文件后缀更改 | `7865` |
| 图片标签编辑器 | `7866` |
| Krea2 提示词反推 | `7867` |
| 新工具建议 | `7870+` |

新增工具时请确保端口没有与已有服务冲突。

---

## 🔐 Local & Privacy

项目默认按本地工具设计：

- 工具服务优先监听 `127.0.0.1`
- 本地模型推理不会因为使用本项目而自动上传到项目作者的服务器
- 第三方网站工具（例如 Danbooru 标签工具）仍需要访问对应第三方服务
- 模型下载、第三方依赖和外部服务可能产生网络请求

使用第三方模型、网站和 Python 依赖时，请遵守对应许可证和服务条款。

---

## 🛠️ Troubleshooting

### 找不到 Python

检查：

```text
python\python.exe
```

项目脚本不会改用系统的 `python` 或 `py`。

如果这是源码仓库，请先准备项目所需的 Python 运行时；如果是 Release 便携包，请重新确认是否完整解压。

### 某个工具启动失败

优先检查：

```text
logs/
```

同时可以直接进入对应工具目录运行 `run.bat`，这样更容易看到完整错误输出。

### Krea2 无法加载模型

确认模型目录和文件完整，并检查 NVIDIA 驱动、PyTorch 与 CUDA 是否正常。

### 标签翻译不可用

确认：

```text
1. llama-cpp-python 已安装
2. Hy-MT2 GGUF 模型存在
3. 当前 Python / CUDA 与 wheel 匹配
```

---

## ✅ Development Checklist

提交新工具前建议检查：

```text
[ ] app.py 可以正常启动
[ ] manifest.json 是合法 JSON
[ ] port 没有冲突
[ ] /ping 返回 200
[ ] 独立 run.bat 可以启动
[ ] 门户可以发现并启动工具
[ ] 工具关闭后没有残留项目进程
[ ] README 已补充工具说明和截图位置
```

---

## 🗺️ Roadmap

后续更倾向于继续加入**小而实用**的 ComfyUI / SD / LoRA 辅助工具，例如：

- Prompt / Danbooru 标签处理
- LoRA / 模型信息分析
- 数据集整理与批处理
- ComfyUI 工作流辅助工具
- 本地模型驱动的小型工具

项目不会强制所有工具使用同一种技术栈；门户层负责统一管理，具体工具保持独立。

---

## 📄 License

本项目采用 [MIT License](LICENSE)。

第三方模型、网站和依赖库分别遵循其自身许可证。
