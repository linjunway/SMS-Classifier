FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SMSCLF_HOME=/app

WORKDIR /app

# 1) Install the package (cached unless dependencies/source change)
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# 2) Train at build time so the model always matches the pinned scikit-learn version
COPY data ./data
RUN python -m smsclf.train

# 3) Run as an unprivileged user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "smsclf.api:app", "--host", "0.0.0.0", "--port", "8000"]
