# Controller service image: ONNX policy + safety shield + MQTT + FastAPI sidecar (CPU only).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir onnxruntime numpy pyyaml pydantic fastapi uvicorn aiomqtt prometheus-client eclipse-sumo libsumo traci sumolib

COPY src /app/src
COPY configs /app/configs
COPY sim/networks /app/sim/networks
COPY sim/vtypes /app/sim/vtypes
COPY models/final /app/models/final

EXPOSE 8080
CMD ["python", "-m", "drs.service.app", "--config", "configs/service.yaml", "--override", "{\"mqtt\": {\"host\": \"mosquitto\"}}"]
