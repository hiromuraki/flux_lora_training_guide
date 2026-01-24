from pathlib import Path
from spandrel import ModelLoader, ImageModelDescriptor
from huggingface_hub import hf_hub_download
import os
import cv2
import torch
import numpy as np
import glob


class ImageUpscaler:
    def __init__(self) -> None:
        repo_id = "ai-forever/Real-ESRGAN"
        model_filename = "RealESRGAN_x4.pth"

        print(f"正在下载/加载模型: {model_filename} ...")
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename
            )
        except Exception as e:
            print(f"下载失败: {e}，请检查网络")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"运行设备: {device}")

        model_loader = ModelLoader(device=device)
        model = model_loader.load_from_file(model_path)

        if not isinstance(model, ImageModelDescriptor):
            raise ValueError("加载的模型不是图像超分模型")

        model.eval()
        if device.type == 'cuda':
            model.cuda()

        self.__model = model
        self.__device = device

    def upscale_image(self, input_dir: Path, output_dir: Path) -> None:
        # 目标分辨率 (1024 是 Flux 的训练标准)
        target_size = 1024

        # 2. 准备输出
        os.makedirs(output_dir, exist_ok=True)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"找到 {len(img_paths)} 张图片，开始处理...")

        for i, path in enumerate(img_paths):
            img_name = os.path.basename(path)
            print(f"[{i+1}/{len(img_paths)}] 处理中: {img_name}")

            try:
                # 使用 numpy 读取二进制流，绕过 OpenCV 的路径解析
                img_stream = np.fromfile(path, dtype=np.uint8)
                img_cv = cv2.imdecode(img_stream, cv2.IMREAD_UNCHANGED)
                if img_cv is None:
                    continue

                # 预处理
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_t = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_t = img_t.to(self.__device)

                # 推理
                with torch.no_grad():
                    # 无论模型是 x2 还是 x4，spandrel 都会自动处理
                    output_t = self.__model(img_t)

                # 后处理
                output_np = output_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_np = np.clip(output_np, 0, 1) * 255.0
                output_np = output_np.astype(np.uint8)
                output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

                # --- 智能调整分辨率 ---
                # 如果用 x4: 512 -> 2048 -> 缩小到 1024 (锐度高，推荐)
                # 如果用 x2: 512 -> 1024 -> 不变 (速度快)
                h, w = output_bgr.shape[:2]

                if h != target_size and w != target_size:
                    scale = target_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    # 使用 INTER_AREA 插值缩小，可以消除噪点，让画面更干净
                    final_img = cv2.resize(output_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    final_img = output_bgr

                # 保存
                # 获取文件扩展名 (例如 .jpg 或 .png)
                save_path = os.path.join(output_dir, img_name)
                file_ext = os.path.splitext(save_path)[1]

                # 将图片编码为二进制流
                success, encoded_img = cv2.imencode(file_ext, final_img)

                # 如果编码成功，使用 numpy 的 tofile 保存
                if success:
                    encoded_img.tofile(save_path)

            except Exception as e:
                print(f"处理失败 {img_name}: {e}")

        print(f"{len(img_paths)} 张图片处理完成")
