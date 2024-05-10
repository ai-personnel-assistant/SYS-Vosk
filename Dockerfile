# Dockerfile

FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

CMD [ "gunicorn", "-w", "4", "-b", "0.0.0.0:3800", "app:app"]

