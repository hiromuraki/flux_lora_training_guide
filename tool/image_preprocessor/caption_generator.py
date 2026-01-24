from volcenginesdkarkruntime import Ark
import json
import os
import sys


class CaptionGenerator:
    def __init__(self) -> None:
        api_key = os.getenv('ARK_API_KEY')
        if not api_key:
            api_key = input("无法在环境变量 ARK_API_KEY 中获取有效的API key，请手动输入API Key：")
            if not api_key:
                sys.exit(1)

        self.__client = Ark(
            base_url='https://ark.cn-beijing.volces.com/api/v3',
            api_key="1661ae96-8124-45cc-a98f-6bfc7836bdb6",
        )

    def generate_captions(self, obj_name: str, class_token: str, base64_images: dict[str, str]) -> dict[str, str] | None:
        messages = [
            {
                "role": "user",
                "content": CaptionGenerator.__get_input_content(
                    base64_images=base64_images,
                    class_token=class_token,
                    obj_name=obj_name,
                ),
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_prompt_pair",
                    "description": "创建用于训练扩散模型的图像文本对中的文本描述文件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt_pair": {
                                "type": "dict[str,str]",
                                "description": "图像 id: prompt字典对"
                            },
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]

        completion = self.__client.chat.completions.create(
            model="doubao-seed-1-6-251015",
            thinking={
                "type": "disabled"
            },
            messages=messages,
            tools=tools)
        resp_msg = completion.choices[0].message

        if completion.choices[0].finish_reason == "tool_calls":
            tool_calls = completion.choices[0].message.tool_calls
            for tool_call in tool_calls:
                if tool_call.function.name == "create_prompt_pair":
                    json_args = json.loads(tool_call.function.arguments)
                    return json_args["prompt_pair"]

        return None

    @classmethod
    def __get_input_content(cls, obj_name: str,  class_token: str, base64_images: dict[str, str]) -> list[dict[str, str]]:
        input_images = [
            {
                "type": "image_url",
                "image_url": image,
            }
            for image in base64_images.values()
        ]
        input_text = {
            "type": "text",
            "text": (
                f"""
                你是一个为 Flux.1 AI 绘图模型准备训练数据的图像标注专家。
                请为提供的 '{obj_name}' 的每张图片写一段详细的自然语言描述（Caption）。
                触发词（TRIGGER_WORD）是{class_token}。
                类型（TYPE）是{obj_name}的大分类，例如蔬菜（vegetable），肉类（meta）等，你需要自行判断。

                请严格遵守以下规则：
                1：**触发词一致**：不得对触发词作任何更改。
                2. **触发词前置**：描述必须以触发词 <TRIGGER_WORD> <TYPE> 开头。
                3. **自然语言**：不要使用标签堆砌（Tag salad），请使用通顺、流畅的英文长句。
                4. **细节描述**：
                - 详细描述 '{obj_name}' 的物理特征（颜色、纹理、形状、根茎叶花的细节）。
                - 描述光影（如自然光、侧光、阴影）。
                - 描述构图和背景（如特写、模糊背景、藤编背景等）。
                5. **格式限制**：直接输出描述文本，不要包含 "Here is the description" 等任何前缀或多余的 Markdown 符号。
                6. **描述词汇**：在 Caption 中，当提到'{obj_name}'时，使用其完整且一致的 TRIGGER_WORD 进行指代。
                
                响应结果以如下格式呈现，不要有任何多余内容\n"
                
                示例格式：
                [00]
                <TRIGGER_WORD> <TYPE>, a close-up photo of [主体描述] featuring [细节特征], under [光照环境] with a [背景描述].
                
                [01]
                <TRIGGER_WORD> <TYPE>, ...

                然后，一致地，根据上述格式调用create_prompt_pair，传递prompt_pair以自动生成文本文件，其中图像 id即上述的形如[00]中的00，your_desc即上述描述，如
                {{ '00': '<TRIGGER_WORD> <TYPE>, your_desc', '01': '<TRIGGER_WORD> <TYPE>, your_desc' }}
                """
            )
        }

        return [
            *input_images,
            input_text,
        ]
