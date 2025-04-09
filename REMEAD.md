## 执行训练

```bash
sudo docker run -it --rm  --gpus=all \
-v ./config.toml:/app/config.toml \
-v ./output:/app/output
-v ~/.cache/huggingface:~/.cache/huggingface \
jtyoui/llm:pytorch2.6-cuda12.4-cudnn9
```

### 格式

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

