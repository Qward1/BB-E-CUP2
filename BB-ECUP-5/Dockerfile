# 1) База
FROM python:3.11-slim

# 2) Настройки окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc make \
    libffi-dev libssl-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip setuptools wheel


COPY requirements.txt /app/requirements.txt

RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt ; \
    else \
        pip install --no-cache-dir fastapi "uvicorn[standard]" ; \
    fi


COPY . /app


RUN mkdir -p /app/workdir/in /app/output \
 && adduser --disabled-password --gecos "" --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000


CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "8000"]
