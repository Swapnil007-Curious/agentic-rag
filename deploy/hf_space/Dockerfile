# Docker Space: the chat assistant on CPU with the hosted model (no local model).
# Built from deploy/hf_space/build/ (python scripts/build_space.py assembles it).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch from the PyTorch CPU index first, then the pinned app requirements
COPY requirements.txt /app/requirements.txt
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /app/requirements.txt

# bge-small (embeddings and the chunk tokenizer) is downloaded while the image is built; the
# running app is offline for models and never downloads one
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5', allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*'])" \
 && chmod -R a+rX /opt/hf-cache

# non-root user (uid 1000, as Spaces expect)
RUN useradd --create-home --uid 1000 user
COPY --chown=user:user . /app
USER user

# public mode: bind 0.0.0.0:7860, rate limits per client behind the Space's proxy, the page may
# be shown in the Space's frame on huggingface.co. LLM_API_KEY comes from the Space secrets.
ENV HOST=0.0.0.0 \
    PORT=7860 \
    TRUST_PROXY=1 \
    PROXY_HOPS=1 \
    FRAME_ANCESTORS=https://huggingface.co \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/app/src

EXPOSE 7860
CMD ["python", "scripts/serve.py", "--threads", "2"]
