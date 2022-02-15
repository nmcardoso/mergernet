FROM python:3-slim

ENV MLFLOW_VERSION 1.23.1
ENV MLFLOW_HOST 0.0.0.0
ENV MLFLOW_PORT 8080
ENV BACKEND_URI sqlite:////app/data/mlflow.sqlite
ENV ARTIFACT_ROOT /app/data/artifacts

WORKDIR /app/
RUN pip install --no-cache-dir mlflow==$MLFLOW_VERSION
EXPOSE 8080

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --serve-artifacts --host ${MLFLOW_HOST} --port ${MLFLOW_PORT}
