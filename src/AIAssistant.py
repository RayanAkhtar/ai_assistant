import os
from queue import Queue

from SpeakerRecorder import SpeakerRecorder
from SpeechToText import AudioTranscriber
from LLMQuery import LLMQuery

recorder = SpeakerRecorder()
transcriber = AudioTranscriber()
llm = LLMQuery()

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
    prev_summary = summary[0]
    transcriptions = summary[1:]

    query = f"""
Your role is to summarise transcripts from the callee in an audio call in order to reduce the number of words/tokens that a summary takes up. You generally do not exceed 500 tokens.
You also preserve key information obtained in the meeting such as the company name, their goals and motivations, as well as any other relevant information that a telemarketer could use to help with the sale of their product.

Here is the current summary:
{prev_summary}

Here are the current transcripts:
{"\n".join(transcriptions)}

Please update the current summary with the current transcriptions to generate a new transcription.
"""
    
    new_summary = llm.generate_query(file_paths=[], few_shot_prompts=[], query=query)
    return [new_summary]


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

def pass_prompt(transcript):
    """Runs the Prompt on an LLM to get a relevant output"""
    # Documents that are to be read must be places in doc/
    # A summary of the current conversation will also be held in doc/


if __name__ == "__main__":
    cleanup()
    transcript = Queue(maxsize=5)
    while True:
        audio_file_path = record_audio()
        transcription = transcribe_audio(audio_file_path)
        latest_sentence = add_to_transcript(transcript, transcription)
        update_summary(latest_sentence)
        message = pass_prompt(transcript)
        print(message)