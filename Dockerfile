FROM python:3.11-slim

WORKDIR /app

# System libraries required by opencv-python (non-headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch before the rest of requirements.txt.
# requirements.txt pins torch==x.x.x+cu121 which only exists on the CUDA
# index; installing CPU torch first and filtering those lines avoids pulling
# multi-GB CUDA wheels into the image.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
       --index-url https://download.pytorch.org/whl/cpu

RUN grep -vE "^torch|^torchvision" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Copy project code last — this layer changes most often and should not
# bust the dependency cache above
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
