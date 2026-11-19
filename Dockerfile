# syntax=docker/dockerfile:1
# CourtSense — hosted image-only demo
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/torch \
    COURTSENSE_DEMO=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip downloads are kept in a BuildKit cache mount, so rebuilds on a slow
# connection don't re-fetch packages that were already downloaded once.
COPY requirements-docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-docker.txt

# Torch last among the pip layers: switching variants only rebuilds from here.
# cu132 (default, NVIDIA GPU, needs host driver R580+) or cpu
ARG TORCH_VARIANT=cu132
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.15.0 torchvision==0.29.1 \
      --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# Bake the COCO-pretrained Faster R-CNN backbone into the image so the
# container starts offline and doesn't re-download on every restart.
RUN python -c "from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights as W; fasterrcnn_resnet50_fpn_v2(weights=W.DEFAULT)"

COPY gui/ gui/
COPY modules/ modules/
COPY utils/ utils/

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8080/_stcore/health || exit 1

CMD ["streamlit", "run", "gui/gui.py"]
