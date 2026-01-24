# FLUX.1 LoRA模型训练指南组

## 1. 步骤说明

1. [注册 AutoDL 账号并创建算力实例](./1.AutoDL注册指南.md)
2. [部署 SD-Trainer 与 ComfyUI](./2.SD-Trainer%20+%20ComfyUI%20部署指南.md)
3. [SD-Trainer 与 ComfyUI 的使用指南](./3.SD-Trainer%20+%20ComfyUI%20训练与验证指南.md)

## 2. 工具说明

1. [训练图像预处理工具](./tool/image_preprocessor/)，用于为原始图像生成 Caption 并进行超分

## 3. 脚本说明

1. [快速通过 tmux 启动 frp + ComfyUI + SD-Trainer](./script/compose-up.sh)
2. [快速同步 SD-Trainer 训练的 LoRA 模型至 ComfyUI](./script/sync-trained-models.sh)

## 3. 附加文件

1. [SD-Trainer FLUX.1 LoRA 训练参数预设](./FLUX.1-LoRA-训练参数.toml)
2. [ComfyUI 带 LoRA 的 FLUX.1 文生图工作流](./flux_dev_with_lora_example.json)
