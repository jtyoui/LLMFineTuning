FROM python:3.12

WORKDIR /app

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt  .

RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124

COPY . .
CMD ["python","main.py"]