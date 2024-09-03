
def cleanup():
    """Removing files in tmp and rec"""
    pass


def record_audio():
    """Records audio from the speaker (for now)"""
    pass

def transcribe_audio():
    """Transcribes the audio file that was just recorded"""
    pass

def add_to_transcript():
    """
    Adds the transcript to the current list of transcriptions
    If the current list size is greater than 5, the oldest files will be removed
    and summarised in a doc file.
    """
    pass

def pass_prompt():
    """Runs the Prompt on an LLM to get a relevant output"""
    pass



if __name__ == "__main__":
    cleanup()
    transcript = []
    while True:
        audio_file = record_audio()
        transcription = transcribe_audio(audio_file)
        transcript = add_to_transcript(transcript, transcription)
        message = pass_prompt()
        print(message)