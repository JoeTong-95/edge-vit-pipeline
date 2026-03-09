FROM python:3.10-bookworm

WORKDIR /app

# install system packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1 \
    libglib2.0-0

# copy requirements
COPY requirements.txt .

# install python packages
RUN pip install --no-cache-dir -r requirements.txt

# copy project
CMD ["bash"]