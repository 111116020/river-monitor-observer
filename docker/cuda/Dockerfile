FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN [ "apt-get", "update" ]
RUN [ "apt-get", "install", "-y", "libglib2.0-0", "libsm6", "libxrender1", "libgl1", "tesseract-ocr" ]
RUN [ "mkdir", "-p", "/workspace" ]

WORKDIR /workspace
COPY models/ /workspace/models/
COPY river_observer/ /workspace/river_observer/
COPY tessdata/ /workspace/tessdata/
COPY config.yaml /workspace
COPY requirements.txt /workspace
RUN [ "python3", "-m", "pip", "install", "-r", "/workspace/requirements.txt" ]

CMD [ "python3", "-m", "river_observer" ]
