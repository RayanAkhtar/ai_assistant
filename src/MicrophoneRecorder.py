import soundcard as sc
import soundfile as sf
import numpy as np
import time
import os
from collections import deque

class MicrophoneRecorder:
    def __init__(self, output_dir="rec", samplerate=40000,
                 max_files=7, silence_threshold=0.01, silence_duration=2.0,
                 max_recording_duration_mins=30):
        self.output_dir = output_dir
        self.samplerate = samplerate
        self.max_files = max_files
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_recording_duration = max_recording_duration_mins * 60

        os.makedirs(self.output_dir, exist_ok=True)

    def is_silence(self, data):
        """Check if the audio data is silent."""
        return np.mean(np.abs(data)) < self.silence_threshold

    def wait_for_sound(self, mic):
        """Wait until sound is detected to start recording."""
        print("Waiting for sound to start recording...")
        while True:
            data = mic.record(numframes=self.samplerate // 10)  # 0.1 secs recorded
            if not self.is_silence(data):
                print("Sound detected! Starting recording...")
                break

    def record_until_silence(self, output_file_name):
        """Record audio from microphone until silence or maximum duration is reached."""
        with sc.default_microphone().recorder(samplerate=self.samplerate) as mic:
            # Wait until sound is detected before starting recording
            self.wait_for_sound(mic)

            frames = []
            start_time = time.time()

            while True:
                data = mic.record(numframes=self.samplerate // 10)  # 0.1 secs recorded
                frames.append(data)

                if self.is_silence(data):
                    silence_start = time.time()
                    while time.time() - silence_start < self.silence_duration:
                        data = mic.record(numframes=self.samplerate // 10)
                        frames.append(data)
                        if not self.is_silence(data):
                            break
                    else:
                        # end of sentence
                        break

                if time.time() - start_time > self.max_recording_duration:
                    print("Maximum recording duration reached.")
                    break

            audio_data = np.concatenate(frames)

            sf.write(file=output_file_name, data=audio_data[:, 0], samplerate=self.samplerate)

    def manage_files(self, file_list):
        """Manage a rotating list of files, keeping only the most recent ones."""
        while len(file_list) > self.max_files:
            oldest_file = file_list.popleft()
            if os.path.exists(oldest_file):
                os.remove(oldest_file)

    def get_file_list(self):
        """Get a list of audio files in the output directory sorted by modification time."""
        files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.endswith('.wav')]
        files.sort(key=os.path.getmtime)  # modification time ordering
        return deque(files)

    def record(self):
        """Main recording loop to record audio from microphone and manage files."""

        try:
            file_list = self.get_file_list()

            current_time = time.strftime("%Y%m%d_%H%M%S")
            output_file_name = os.path.join(self.output_dir, f"recording_{current_time}.wav")

            self.record_until_silence(output_file_name)
            file_list.append(output_file_name)

        except KeyboardInterrupt:
            print("Recording stopped by user.")

if __name__ == "__main__":
    recorder = MicrophoneRecorder()
    recorder.record()
