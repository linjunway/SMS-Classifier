FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY app/requirements.txt .
RUN pip install -r requirements.txt
COPY app/streamlit_app.py .
RUN useradd --create-home appuser
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
