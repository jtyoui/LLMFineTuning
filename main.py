import os
import pprint
import tomllib
from os import path

import pandas as pd

llm = os.environ.get("LLM", "/llm")

with open(path.join(llm, "config.toml"), "rb") as f:
    config = tomllib.load(f)

if "Envs" in config:
    for k, v in config['Envs'].items():
        os.environ[k] = str(v)

# 重写路径
if not config["Pretrained"]["model_name"].startswith("/"):
    config["Pretrained"]["model_name"] = path.join(llm, config["Pretrained"]["model_name"])

if not config['SFTConfig']['output_dir'].startswith("/"):
    config["SFTConfig"]["output_dir"] = path.join(llm, config["SFTConfig"]["output_dir"])

if not config['args']['data_path'].startswith("/"):
    config['args']['data_path'] = path.join(llm, config['args']['data_path'])

# 如果策略是测试，那么加载的模型来自于output_dir
if config['args']['strategy'] == 'test':
    config['Pretrained']['model_name'] = config['SFTConfig']['output_dir']

pprint.pprint(config, indent=2)

data = config['args']['data_path']
suffix = os.path.splitext(data)[-1]

if suffix == '.xlsx':
    df = pd.read_excel(data)
elif suffix == '.csv':
    df = pd.read_csv(data)
elif suffix == '.json':
    df = pd.read_json(data)
elif suffix == '.parquet':
    df = pd.read_parquet(data)
else:
    raise ValueError("无效的数据类型")

if __name__ == '__main__':
    ## 这行要放在加载环境变量后面
    from llm import FineTuning

    model = FineTuning(config)
    if config['args']['strategy'] == "train":
        model.train(df)
    elif config['args']['strategy'] == "test":
        model.evaluate(df)
