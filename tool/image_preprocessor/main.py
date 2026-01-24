import base64
import shutil
import sys
from pypinyin import lazy_pinyin
from pathlib import Path
from image_upscaler import ImageUpscaler
from caption_generator import CaptionGenerator

DATASET_DIR = Path("./dataset")
OUTPUT_DIR = Path("./train")
ACCEPTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def convert_to_class_token(obj_name: str) -> str:
    pinyin_list = lazy_pinyin(obj_name)
    return "".join(pinyin_list).upper()


def get_image_base64_url(image_file: Path) -> str:
    with open(image_file, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
        img_ext = Path(image_file).suffix.lower()
        if img_ext == ".jpg" or img_ext == ".jpeg":
            return f"data:image/jpg;base64,{img_base64}"
        elif img_ext == ".png":
            return f"data:image/png;base64,{img_base64}"
        else:
            raise KeyError(f"不支持的图像格式：{img_ext}")


image_dirs = [x for x in DATASET_DIR.iterdir() if x.is_dir()]
caption_generator = CaptionGenerator()
image_upscaler = ImageUpscaler()

for image_dir in image_dirs:
    obj_name = image_dir.stem
    class_token = convert_to_class_token(obj_name)

    print(f"正在处理: {obj_name} ({class_token})")

    # 1. 规范化文件名，全部以两位数字命名（假设训练图像均不超过100张），以保证有序性
    filenames = [x.absolute() for x in image_dir.iterdir() if x.is_file()
                 and x.suffix.lower() in ACCEPTED_IMAGE_EXTS]
    img_id = 0
    image_files: list[Path] = []
    for filename in filenames:
        normalized_path = image_dir / f"{img_id:02}{filename.suffix}"
        shutil.move(filename, normalized_path)
        image_files.append(normalized_path)
        img_id += 1

    # 2. 生成 caption
    base64_images = {image_file.stem: get_image_base64_url(image_file) for image_file in image_files}
    image_captions = caption_generator.generate_captions(obj_name, class_token, base64_images)
    if image_captions is None:
        print(f"无法获取'{obj_name}'的 Caption")
        sys.exit(1)

    train_dir = OUTPUT_DIR / image_dir.stem / f"5_{image_dir.stem}"
    train_dir.mkdir(parents=True, exist_ok=True)
    print(image_captions)
    for img_id, caption in image_captions.items():
        with open(train_dir / f"{img_id}.txt", mode="w", encoding="utf8") as fp:
            fp.write(caption)

    # 3. 生成超分图
    image_upscaler.upscale_image(image_dir, train_dir)
