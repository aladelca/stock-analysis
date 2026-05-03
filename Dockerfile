FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN uv sync --frozen --extra gcp --extra mlflow --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["stock-analysis"]
CMD ["run-gcp-one-shot", "--config", "configs/portfolio.gcp.yaml", "--forecast-engine", "ml"]
