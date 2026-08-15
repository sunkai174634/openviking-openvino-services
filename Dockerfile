FROM openvino/ubuntu22_runtime:2026.3.0
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv/openviking-openvino
COPY wheels /tmp/wheels
RUN python3 -m pip install --no-index --find-links=/tmp/wheels \
    fastapi==0.125.0 \
    "uvicorn[standard]==0.38.0" \
    pydantic==2.13.0 \
    transformers==5.2.0 \
    optimum==2.3.0 \
    optimum-intel==2.1.0 \
    tokenizers==0.23.0 \
    sentencepiece \
    huggingface_hub==1.3.5 \
    safetensors==0.7.0 \
    accelerate==1.12.0 \
    && rm -rf /tmp/wheels
CMD ["python3", "-c", "print('openviking-openvino base image ready')"]
