# train: everything needed to run jobs and the MLflow server
FROM python:3.12-slim AS train
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[train]"
# first-run seed, copied onto the data volume by the `seed` job (a mounted volume would shadow /data)
COPY data/rates.csv /seed/rates.csv
ENV VOLOPS_STORE=/data/rates.csv VOLOPS_MONITOR=/data/monitor.json

# serve: only the slim runtime (mlflow-skinny), no training stack, non-root
FROM python:3.12-slim AS serve
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "volops.serve:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
