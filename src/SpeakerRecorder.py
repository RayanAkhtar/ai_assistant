# audio_recorder.py

import soundcard as sc
import soundfile as sf
import numpy as np
import time
import os
from collections import deque

class SpeakerRecorder:
    def __init__(self, output_dir="rec", samplerate=40000, record_duration=5,
                 max_files=7, silence_threshold=0.01, silence_duration=2.0,
                 max_recording_duration=60):
        self.output_dir = output_dir
        self.samplerate = samplerate
        self.record_duration = record_duration
        self.max_files = max_files
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_recording_duration = max_recording_duration

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def is_silence(self, data):
        """Check if the audio data is silent."""
        return np.mean(np.abs(data)) < self.silence_threshold

    def record_until_silence(self, output_file_name):
        """Record audio until silence or maximum duration is reached."""
        with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=self.samplerate) as mic:
            frames = []
            start_time = time.time()

            while True:
                # Record a chunk of audio
                data = mic.record(numframes=self.samplerate // 10)  # Record 0.1 seconds of audio
                frames.append(data)

                # Flatten audio data for silence detection
                flat_data = np.concatenate(frames)

                # Check for silence
                if self.is_silence(data):
                    silence_start = time.time()
                    while time.time() - silence_start < self.silence_duration:
                        data = mic.record(numframes=self.samplerate // 10)
                        frames.append(data)
                        if not self.is_silence(data):
                            # Reset silence detection if voice is detected
                            break
                    else:
                        # Silence detected for specified duration
                        print("End of sentence detected.")
                        break

                # Check for maximum recording duration to prevent infinite loop
                if time.time() - start_time > self.max_recording_duration:
                    print("Maximum recording duration reached.")
                    break

            # Concatenate all frames
            audio_data = np.concatenate(frames)

            # Save the recorded audio to a file
            sf.write(file=output_file_name, data=audio_data[:, 0], samplerate=self.samplerate)
            print(f"Recording saved as {output_file_name}")

    def manage_files(self, file_list):
        """Manage a rotating list of files, keeping only the most recent ones."""
        while len(file_list) > self.max_files:
            print("time to remove some files")
            oldest_file = file_list.popleft()
            if os.path.exists(oldest_file):
                os.remove(oldest_file)
                print(f"Deleted old file: {oldest_file}")

    def get_file_list(self):
        """Get a list of audio files in the output directory sorted by modification time."""
        files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.endswith('.wav')]
        files.sort(key=os.path.getmtime)  # Sort files by modification time
        return deque(files)

    def record(self):
        """Main recording loop to record audio and manage files."""

        while True:
            try:
                file_list = self.get_file_list()  # Initialize the file list with existing files

                current_time = time.strftime("%Y%m%d_%H%M%S")
                output_file_name = os.path.join(self.output_dir, f"recording_{current_time}.wav")

                # Record audio until silence or max duration is reached
                self.record_until_silence(output_file_name)

                # Add the new file to the list
                file_list.append(output_file_name)

                # Manage the list to keep only the most recent files
                self.manage_files(file_list)

            except KeyboardInterrupt:
                print("Recording stopped by user.")
                break

if __name__ == "__main__":
    recorder = SpeakerRecorder()
    recorder.record()
