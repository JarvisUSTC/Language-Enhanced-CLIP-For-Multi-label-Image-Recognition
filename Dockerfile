FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /workspace

COPY ./requirements.txt ./requirements.txt
COPY ./test.sh ./test.sh
COPY ./train_1.sh ./train_1.sh
COPY ./train_2.sh ./train_2.sh
RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx libglib2.0-dev

RUN pip install -r requirements.txt \
    && pip install -U openmim \
    && mim install mmcv-full==1.7.0 \
    && python -c "import nltk; nltk.download('all')"
