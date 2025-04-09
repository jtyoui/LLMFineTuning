FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV LLM /llm

COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124

CMD ["python","main.py"]