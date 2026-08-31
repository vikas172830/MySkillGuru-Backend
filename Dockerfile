
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install --with-deps chromium

COPY . .

EXPOSE 5051

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5051"]
