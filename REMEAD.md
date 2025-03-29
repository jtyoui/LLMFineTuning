## 执行训练

```bash
sudo docker run -it -v .:/app --gpus=all -v ~/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B:/app/pretrained llm
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

