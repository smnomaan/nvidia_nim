FROM python:3.11-slim

ARG UV_VERSION=0.8.15
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Install locked dependencies before copying frequently changed application code.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev

COPY app.py main.py ./
COPY src ./src

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/chroma_db /app/state /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache

USER appuser
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
