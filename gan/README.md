# 环境配置
- conda 创建环境环境 如 :<br>
conda create -n deeplearn python==3.8
- 并激活环境 conda activate deeplearn
- deeplearn 环境下 安装torch gpu版本<br>
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
- deeplearn 环境下安装其他包<br>
在当前目录下： pip install -r requirements.txt