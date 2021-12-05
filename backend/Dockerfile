FROM nvidia/cuda:10.0-base

WORKDIR /app

RUN apt-get update && apt-get install &&\
apt-get install git &&\
git clone https://github.com/kingoflolz/CLIP_JAX &&\
git clone https://github.com/nshepperd/jaxtorch

COPY . . 

RUN pip install -r requirements.txt

EXPOSE 5000
