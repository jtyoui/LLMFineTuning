import os
import pprint
import tomllib

import pandas as pd

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

if "Envs" in config:
    for k, v in config['Envs'].items():
        os.environ[k] = str(v)

pprint.pprint(config)

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
