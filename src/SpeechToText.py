from openai import OpenAI
import os
from dotenv import load_dotenv


class AudioTranscriber:
    def __init__(self):
        """
        Initialize the AudioTranscriber with the specified model type.

        :param model_type: The Whisper model type to use. Default is 'base'.
        """
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)

    def transcribe(self, audio_path):
        """
        Transcribe the given audio file.

        :param audio_path: Path to the audio file to transcribe.
        :return: The transcribed text.
        """
        audio_file = open(audio_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text
        

    def save_transcription(self, transcription, output_path):
        """
        Save the transcription to a text file.

        :param transcription: The transcribed text to save.
        :param output_path: The path where the transcription will be saved.
        """
        with open(output_path, "w") as f:
            f.write(transcription)
        print(f"Transcription saved to {output_path}")


def get_files_sorted_by_creation_time(directory):
    """
    Get a list of files in a directory, sorted by creation time.

    :param directory: Path to the directory to scan.
    :return: A list of file paths, sorted by creation time.
    """
    files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    
    files.sort(key=lambda x: os.path.getctime(x)) # Creation time
    
    return files

if __name__ == "__main__":
    directory = "tst"
    transcriber = AudioTranscriber()
    files = get_files_sorted_by_creation_time(directory)
    transcription = ""
    for file in files:
        if file.split('.')[-1] == "wav":
            transcription += transcriber.transcribe(file).text + "\n"
    print("Transcription:\n", transcription)

