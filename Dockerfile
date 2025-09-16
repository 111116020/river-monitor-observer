FROM intel/intel-extension-for-pytorch:2.8.10-xpu

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libgl1 tesseract-ocr
RUN mkdir /workspace

WORKDIR /workspace
COPY river_observer/ /workspace/river_observer
COPY requirements.txt /workspace
RUN python3 -m pip install -r /workspace/requirements.txt
CMD python3 /workspace/river_observer
