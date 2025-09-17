FROM ultralytics/ultralytics:latest

RUN [ "mkdir", "-p", "/workspace" ]

WORKDIR /workspace
COPY models/ /workspace/models/
COPY river_observer/ /workspace/river_observer/
COPY tessdata/ /workspace/tessdata/
COPY config.yaml /workspace
COPY requirements.txt /workspace
RUN [ "python3", "-m", "pip", "install", "-r", "/workspace/requirements.txt" ]

CMD [ "python3", "-m", "river_observer" ]
