FROM europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

WORKDIR /musicnet
COPY . /musicnet

RUN pip install librosa mido dvclive pandas
RUN apt-get update -y
RUN apt-get install -y fluidsynth

ENV PYTHONPATH=/musicnet

ENTRYPOINT ["musicnet/models/transformer/pipeline.sh"]