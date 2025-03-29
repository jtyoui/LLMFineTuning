#!/bin/bash

sudo docker run --rm -it -v .:/app --gpus=all llm
