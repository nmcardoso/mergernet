FROM python:3-slim

WORKDIR /app/
RUN pip install --no-cache-dir mlflow==$MLFLOW_VERSION
EXPOSE 8080

# ENV BACKEND_URI sqlite:////mlflow/mlflow.db
# ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --serve-artifacts --host ${MLFLOW_HOST} --port ${MLFLOW_PORT}
