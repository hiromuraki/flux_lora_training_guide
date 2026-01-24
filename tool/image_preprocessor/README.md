# 使用方法

## === 已过时，等待重写 ===

## 1. 安装 uv

可直接使用 `pip` 安装

```bash
pip install uv
```

## 2. 将图像文件按如下目录放置

示例

```txt
项目根目录
    /dataset/
        /空心菜/
            图像1.jpg
            图像2.jpg
            图像3.jpg
        /胡萝卜/
            图像1.jpg
            图像2.jpg
            图像3.jpg
    main.py
    ...
```

> 仅支持 `jpg` 与 `png` 格式的图像文件

## 3. 运行主程序，等待自动生成图像的对应的提示词

运行程序

```bash
uv run main.py
```

生成结果示例

```txt
/dataset/
    /空心菜/
        00.jpg
        00.txt
        01.jpg
        01.txt
        02.jpg
        02.txt
    /胡萝卜/
        00.jpg
        00.txt
        01.jpg
        01.txt
        02.jpg
        02.txt
```

> 注1：所有图像的文件名均会被修改为形如 `00`、`01` 的格式
> 注2：为了调用大模型，你需要一个火山方舟大模型服务平台的 `API Key`