FROM python:3.12-slim

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV PYTHONPATH=/service
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 4 workers; tune to (2 × CPU cores) + 1 in production
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--access-log"]
