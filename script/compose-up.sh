#!/bin/bash

# 启动 frp
tmux new -d -s frp "cd /root/ai_frp && ./frpc --config ./frpc.toml"

# 启动 SD-Trainer
tmux new -d -s sd-trainer "cd /root/lora-scripts && uv run gui.py"

# 启动 ComfyUI
tmux new -d -s comfyui "cd /root/ComfyUI && uv run main.py"
