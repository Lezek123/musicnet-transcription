FROM europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

WORKDIR /musicnet

RUN pip install librosa mido dvclive pandas seaborn
RUN apt-get update -y
RUN apt-get install -y fluidsynth
RUN git config --global user.name "Google Cloud"
RUN git config --global user.email "gc@example.com"

ENV PYTHONPATH=/musicnet
ENV MN_DS_PATH=/gcs/musicnet-ds/MusicNet
# ENTRYPOINT ["musicnet/models/transformer/pipeline.sh"]