FROM python:3.12-slim
WORKDIR /app
COPY ml/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY ml /app/ml
RUN mkdir -p /app/models
ENV MODEL_PATH=/app/models/harmony_detector_best.pth
ENV SCORE_THRESHOLD=0.50
EXPOSE 8000
CMD ["uvicorn","ml.serve:app","--host","0.0.0.0","--port","8000"]
