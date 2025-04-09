import os

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

llm = os.environ["LLM"]

class FineTuning:
    def __init__(self, config):
        self.config = config
        model, tokenizer = FastLanguageModel.from_pretrained(**self.config['Pretrained'])
        if self.config['args']['strategy'] == 'train':
            model = FastLanguageModel.get_peft_model(model=model, **self.config['Peft'])
        self.model = model
        self.tokenizer = tokenizer
        self.config['SFTConfig']['max_seq_length'] = self.config['Pretrained']['max_seq_length']
        self.args = SFTConfig(**self.config['SFTConfig'])

    def train(self, df: pd.DataFrame):
        eos = self.tokenizer.eos_token
        dataset = Dataset.from_pandas(df)

        column = self.config['args']['train_column']

        dataset = dataset.map(lambda x: {"text": [i + eos for i in x[column]]}, batched=True,remove_columns=df.columns.tolist())

        print(dataset["text"][0])

        trainer = SFTTrainer(model=self.model, processing_class=self.tokenizer, train_dataset=dataset, args=self.args)
        trainer.train()
        save_model = os.path.join(llm, self.config['SFTConfig']['output_dir'])
        self.model.save_pretrained(save_model)
        self.tokenizer.save_pretrained(save_model)

    def evaluate(self, df: pd.DataFrame):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FastLanguageModel.for_inference(self.model.to(device))

        column = self.config['args']['train_column']
        max_new_tokens = self.config['args']['max_new_tokens']
        save = self.config['args']['data_path']

        results = []
        for text in tqdm(df[column].values):
            tokens = self.tokenizer(text, return_tensors="pt").to(device)
            predict = model.generate(**tokens, max_new_tokens=max_new_tokens, use_cache=True)
            decode = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]
            result = decode[len(text):].strip()
            results.append(result)
        df["LLM Predict"] = results
        df.to_excel(save, index=False)
