FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY config ./config
RUN mkdir -p /app/data
RUN useradd -r -u 10001 pulseiq && chown -R pulseiq /app
USER pulseiq
EXPOSE 8092
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8092/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--http", "h11", "--host", "0.0.0.0", "--port", "8092"]
