import grpc
from tts_pb2 import TextRequest
from tts_pb2_grpc import TextToSpeechStub

def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = TextToSpeechStub(channel)

    text_prompt = "Hello, this is a test to determine if our model is working"
    request = TextRequest(text=text_prompt)

    response = stub.ConvertTextToSpeech(request)
    audio_data = response.audio

    # Handle the audio data (e.g., save it as a WAV file or play it)

if __name__ == '__main__':
    main()
