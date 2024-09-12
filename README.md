# AI-Powered Virtual Assistant
An AI assistant designed to provided advice for the caller, for use in online meetings.

## Requirements
To run this program, you will need to have python installed, and also import multiple packages. The console output should state which packages are required and how to install them if you do not currently have them downloaded.

## Before use
Please ensure that the documents you want the AI assistant to read from are stored in `doc/`. If the folder doesn't exist, please create it.

## How to use
Run the program using `python [path to the folder]/src/AIAssistant.py`. This will then begin the program, feel free to speak into the microphone, or pass through computer audio to generate LLM responses.

## Recording Audio
As of the current version, this program will record any audio played out through your device's speaker. It will record it at the exact audio, so the higher your volume, the louder the recording. If the callee audio is not being recorded properly, consider turning up the volume.