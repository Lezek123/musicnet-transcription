from .base import BaseTrack
import subprocess
import os

class ConvertableMidiTrack(BaseTrack):
    def generate_wav(self):
        midi_path = self.get_midi_path()
        wav_path = self.get_wav_path()
        subprocess.run(
            f"fluidsynth -F {wav_path} /usr/share/sounds/sf2/default-GM.sf2 {midi_path}",
            shell=True,
            capture_output=True
        )
        if self.get_duration() < 1:
            os.remove(wav_path)
            raise Exception("Error generating the WAV")