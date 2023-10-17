import grpc
from tts_pb2 import TextRequest
from tts_pb2_grpc import TextToSpeechStub
import logging

def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = TextToSpeechStub(channel)
    while True:
        text_prompt = input("Enter a text prompt (or type 'exit' to quit): ")

        if text_prompt.lower()=="exit":
            break
        request = TextRequest(text=text_prompt)

    response = stub.ConvertTextToSpeech(request)
    audio_data = response.audio

    logging.info(audio_data)

    # Handle the audio data (e.g., save it as a WAV file or play it)

if __name__ == '__main__':
    main()
