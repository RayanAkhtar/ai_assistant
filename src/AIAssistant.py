import os
from queue import Queue

from SpeakerRecorder import SpeakerRecorder
from SpeechToText import AudioTranscriber


recorder = SpeakerRecorder()
transcriber = AudioTranscriber()

transcript_summary_path = "doc/summary.txt"

def cleanup():
    """Removing files in tmp and rec"""
    # For now, it only ignores tmp.txt, this part will be removed upon completing the AI Assistant
    directories_to_clean = ['tmp', 'rec']
    for directory in directories_to_clean:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename != 'tmp.txt':
                os.remove(file_path)

def record_audio():
    """Records audio from the speaker (for now)"""
    return recorder.record() # likely to change when integrating microphone recording

def transcribe_audio(audio_file_path):
    """Transcribes the audio file that was just recorded"""
    return transcriber.transcribe(audio_file_path)

def add_to_transcript(transcript: Queue, transcription):
    """
    Adds the transcript to the current list of transcriptions
    If the current list size is greater than maxsize, the function will return the recently popped sentence
    """
    latest_sentence = ""
    if transcript.full():
        latest_sentence = transcript.get()
    
    transcript.put(transcription)
    return latest_sentence
    
def summarise(summary: list[str]):
    "Summarises the summary by updating the first index(summary) with the following indices(transcriptions)"
    pass # todo

def update_summary(latest_sentence):
    """Updates the documented summary of the transcript with the latest sentence"""
    summary: list[str] = ""

    with open(transcript_summary_path, "r") as fr:
        summary = fr.readlines()
        max_transcriptions = 6 # first index is summary, rest is transcriptions
        
        if summary.count > max_transcriptions:
            summary = summarise(summary)
        
        fr.close()

    with open(transcript_summary_path, "w") as fw:
        summary.append(latest_sentence)
        fw.writelines(summary)    

def pass_prompt():
    """Runs the Prompt on an LLM to get a relevant output"""
    pass # todo

if __name__ == "__main__":
    cleanup()
    transcript = Queue(maxsize=5)
    while True:
        audio_file_path = record_audio()
        transcription = transcribe_audio(audio_file_path)
        latest_sentence = add_to_transcript(transcript, transcription)
        update_summary(latest_sentence)
        message = pass_prompt()
        print(message)