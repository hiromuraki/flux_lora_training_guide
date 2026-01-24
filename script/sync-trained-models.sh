#!/bin/bash

lora_output_dir="/root/lora-scripts/output"
comfyui_lora_dir="/root/ComfyUI/models/loras"

# 创建目标目录（如果不存在）
mkdir -p "$comfyui_lora_dir"

# 遍历lora输出目录中的所有文件
find "$lora_output_dir" -maxdepth 1 -type f -name "*.safetensors" | while read -r lora_file; do
    filename=$(basename "$lora_file")
    
    # 检查文件名是否符合模式：不以 "-数字.safetensors" 结尾
    if [[ ! "$filename" =~ -[0-9]+\.safetensors$ ]]; then
        source_path="$lora_output_dir/$filename"
        link_path="$comfyui_lora_dir/$filename"
        
        # 如果软链接已存在
        if [[ -L "$link_path" ]]; then
            current_target=$(readlink -f "$link_path" 2>/dev/null)
            if [[ "$current_target" != "$source_path" ]]; then
                rm -f "$link_path"
                ln -sf "$source_path" "$link_path"
            fi
        else
            # 如果软链接不存在，创建新链接
            ln -sf "$source_path" "$link_path"
        fi
    fi
done

# 清理失效的软链接
find "$comfyui_lora_dir" -maxdepth 1 -type l ! -exec test -e {} \; -exec rm {} +