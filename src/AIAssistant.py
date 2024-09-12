import warnings
import os
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import threading
from fpdf import FPDF

from Recorder import Recorder
from SpeechToText import AudioTranscriber
from LLMQuery import LLMQuery

warnings.filterwarnings("ignore", message=".*data discontinuity.*")

speaker_recorder = Recorder(mode='speaker')
microphone_recorder = Recorder(mode='microphone')
transcriber = AudioTranscriber()
llm = LLMQuery()

transcript_summary_path = "doc/summary.txt"
transcript = Queue(maxsize=5)

transcript_lock = threading.Lock()
file_lock = threading.Lock()

debug = True

def cleanup():
    """
    Cleans up the past session data by removing the data stores in tmp/, rec/ and debug/
    """
    directories_to_clean = ['tmp', 'rec', 'debug']
    for directory in directories_to_clean:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    if os.path.exists(transcript_summary_path):
        os.remove(transcript_summary_path)

def transcribe_audio(audio_file_path):
    """
    Transcribes the audio that was just recorded.

    :param audio_file_path: The path to the audio file
    :return: A transcription for that audio file
    """
    print("Transcribing audio...")
    return transcriber.transcribe(audio_file_path)

def add_to_transcript(transcript: Queue, transcription):
    """
    Adds the transcription to the transcript, maintaining the maximum number of items allowed in the transcription.

    :param transcript: A queue that contains each transcription.
    :param transcription: The next transcription to add to the queue.

    :return: The transcription that was removed (if any)
    """
    latest_sentence = ""

    with transcript_lock:
        if transcript.full():
            latest_sentence = transcript.get()
        
        transcript.put(transcription)
    return latest_sentence
    
def summarise(summary: list[str]):
    """
    Summarises the summary by combining latter indices into the first one
    
    :param summary: The summary you wish to summarise.
        summary[0]: The summary
        summary[1..]: Any transcriptions you wish to update the summary with.
    :return: The updated summary
    """
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
    
    print("Generating summary update...")
    new_summary = llm.generate_query(file_paths=[], few_shot_prompts=[], query=query)
    return [new_summary]

def update_summary(latest_sentence):
    """
    Updates the summary with the most recent transcription

    :param latest_sentence: The sentence that was removed when updating the transcription
    """    
    summary: list[str] = []

    try:
        with open(transcript_summary_path, "r") as fr:
            summary = fr.readlines()
            max_transcriptions = 6  # first index is summary, rest is transcriptions
            
            if len(summary) > max_transcriptions:
                summary = summarise(summary)
    except FileNotFoundError:
        summary = []

    with open(transcript_summary_path, "w") as fw:
        summary.append("\n" + latest_sentence)
        fw.writelines(summary) 

def write_to_eof(pathname, text_to_write):
    with file_lock:
        with open(pathname, 'a') as file:
            file.write(text_to_write)
            file.write('\n')

def get_files_in(directory, ignored_files):
    """
    Get a list of files in a directory, sorted by creation time.

    :param directory: Path to the directory to scan.
    :param ignored_files: A list of files to ignore in the scan.

    :return: A list of file paths, sorted by creation time.
    """
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename not in ignored_files:
                files.append(os.path.join(root, filename))
    return files

def pass_prompt(transcript: Queue):
    """
    Runs the prompt on the llm to generate responses for the caller

    :param transcript: The current transcript of the call
    :return: The output for the caller
    """
    file_paths = get_files_in(directory="doc/", ignored_files=["tmp.txt"])
    try:
        summary: list[str] = []
        with open(transcript_summary_path, "r") as fr:
            summary = fr.readlines()

    except FileNotFoundError:
        summary = []

    query = f"""
Your role is a caller making a call to a company, the purpose is to sell a product/service to the other party.
You can only use stats and figures that are given in the context provided.
If you give stats and figures, you explain how you obtained them.
You do not hallucinate data.
You do not give generic answers.
You say 'void' if you cannot find anything relevant to say

---------------------
Here is a summary of the current conversation:
{summary}

---------------------

Here is the transcript between the caller and callee:
{"\n".join(list(transcript.queue))}

---------------------

Please suggest up to 3 possible sentences that could help increase the chances of this call going well, do not number these sentences.
Only write sentences that use information that is specific to either the call or the context provided.
Make sure your output is short, snappy and to the point, writing each point on a new line.
Make sure the sentences you suggest are relevant to the previous point the callee has mentioned.
"""
    
    print("Generating query...")
    output = llm.generate_query(
        file_paths,
        [],
        query
    )
    return output

def ai_assistant_loop():
    """
    The 'main loop' of the ai assistant, records from the speaker, and after each pause,
    an llm will be prompted to make suggestions for the caller.
    """
    while True:
        try:
            print("Listening for audio from speaker...")
            audio_file_path = speaker_recorder.record()
            
            print("Transcribing speaker audio...")
            transcription = f"(CALLEE) - {transcriber.transcribe(audio_file_path)}"
            if debug:
                debug_txt = f"{audio_file_path} {transcription}"
                write_to_eof("debug/transcript.txt", debug_txt)

            latest_sentence = add_to_transcript(transcript, transcription)
            if latest_sentence != "":
                update_summary(latest_sentence)

            print("Generating response...")
            os.system('cls' if os.name == 'nt' else 'clear')
            message = pass_prompt(transcript)
            print(message)
        except KeyboardInterrupt:
            print("Program interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Error in the speaker recording loop: {e}. Continuing...")
            continue
        
def microphone_recording_loop():
    """
    The background loop for the ai assistant. Takes input from the microphone so that the
    responses generated from the assistant will be more tailored to that call session.
    """
    while True:
        try:
            print("Listening for audio from microphone...")
            audio_file_path = microphone_recorder.record()

            print("Transcribing microphone audio...")
            transcription = f"(CALLER) - {transcriber.transcribe(audio_file_path)}"
            if debug:
                debug_txt = f"{audio_file_path} {transcription}"
                write_to_eof("debug/transcript.txt", debug_txt)
            
            latest_transcription = add_to_transcript(transcript, transcription)
            if latest_transcription != "":
                update_summary(latest_transcription)
            
        except KeyboardInterrupt:
            print("Program interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Error in microphone recording loop: {e}. Continuing...")
            continue

if __name__ == "__main__":
    cleanup()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(microphone_recording_loop)
            ai_assistant_loop()
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        speaker_recorder.stop()
        microphone_recorder.stop()
        print("Cleanup done. Program terminated.")
