FROM europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

WORKDIR /

COPY .git-credentials /root/.git-credentials
RUN git config --global credential.helper store
RUN pip install librosa mido dvclive pandas seaborn
RUN apt-get update -y
RUN apt-get install -y fluidsynth
RUN git config --global user.name "Google Cloud"
RUN git config --global user.email "gc@example.com"

ENV PYTHONPATH=/musicnet
ENV MN_DS_PATH=/gcs/musicnet-ds/MusicNet

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["git clone https://github.com/Lezek123/musicnet-transcription.git musicnet && ./musicnet/scripts/gc_run.sh"]