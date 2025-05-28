# Qwen2.5 Omni实时API演示项目

本项目展示如何使用Qwen2.5 Omni模型的实时API进行语音交互。包含完整的语音识别、处理和合成功能。

## 项目特点
- 实时语音交互
- 支持多种语音模式（Chelsie, Serena, Ethan, Cherry）
- 集成Gradio和FastAPI框架
- 使用WebSocket进行实时通信

## 安装依赖
```bash
pip install -r requirements.txt
```

## 运行项目
1. 设置环境变量：
```bash
export MODE=UI
```
2. 运行项目：
```bash
python app.py
```

## 配置
在app.py中设置你的API密钥：
```python
API_KEY = "Your_API_KEY"
```

## 文件结构
- `app.py` - 主程序文件
- `README.md` - 项目文档
- `requirements.txt` - 依赖列表