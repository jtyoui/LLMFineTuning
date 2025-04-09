## 新创建自己的文件夹

    里面要包括这些东西
    1. 自己的训练集
    2. 预训练大模型
    3. 配置文件

## 配置文件看

[config.toml](config.toml)

## 训练集格式

```json
[
  {
    "text": "1.xxx"
  },
  {
    "text": "2.xxx"
  },
  {
    "text": "3.xxx"
  }
]

```

## 执行训练

```bash
sudo docker run -it --gpus=all -v {文件夹路径}:/llm --name llm -d jtyoui/llm:pytorch2.6-cuda12.4-cudnn9
```
