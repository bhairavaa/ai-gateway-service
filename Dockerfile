# --- Build stage: install deps into an isolated prefix, keep build tools out of the final image ---
FROM python:3.13-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
FROM python:3.13-slim

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations

# SQLite file lives here, mounted as a volume in docker-compose so data survives rebuilds.
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

# Apply any pending migrations before serving — the data volume may be from an older image.
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000
