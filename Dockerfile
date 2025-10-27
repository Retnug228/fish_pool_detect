FROM ultralytics/ultralytics:latest-arm64

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir lap>=0.5.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
RUN mkdir -p csv

CMD ["python3", "detectForKhadas.py"]