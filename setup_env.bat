@echo off
rem ============================================================
rem 统一启动器 - Python 环境一键安装
rem 适用于全新机器：只要本文件夹完整（含 python\ 内置解释器），
rem 双击本脚本即可装齐所有依赖。需要联网。
rem ============================================================
cd /d "%~dp0"

if not exist "python\python.exe" (
    echo [错误] 找不到内置解释器 python\python.exe
    echo 请确认整个"统一启动器"文件夹完整复制，不要单独抽出一个文件。
    pause
    exit /b 1
)

echo.
echo [1/3] 安装 PyTorch（CUDA 12.8 版，约 3GB，请耐心等待）...
python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

echo.
echo [2/3] 安装 llama-cpp-python（图片标签编辑器的离线翻译引擎）...
if exist "llama_cpp_python-0.3.49+cu128-cp312-cp312-win_amd64.whl" (
    python\python.exe -m pip install "llama_cpp_python-0.3.49+cu128-cp312-cp312-win_amd64.whl"
) else (
    echo   未找到本地 GPU 版 whl（文件较大不随仓库分发），回退安装 CPU 版...
    echo   翻译功能可用但速度较慢；如需 GPU 加速请见 README.md "GPU 翻译加速"一节。
    python\python.exe -m pip install llama-cpp-python
)
if errorlevel 1 goto :fail

echo.
echo [3/3] 安装其余依赖...
python\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo 验证安装...
python\python.exe -c "import torch, transformers, flask, llama_cpp, safetensors, numpy, PIL; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available()); print('全部依赖就绪！')"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 环境安装完成！双击 run.bat 启动统一门户。
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [错误] 安装失败，请检查网络后重试，或把上面的报错发给开发者。
pause
exit /b 1
