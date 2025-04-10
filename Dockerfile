FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY . .

RUN apt update && apt install -y build-essential

RUN pip3 install -r requirements.txt
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124

CMD ["python","main.py"]