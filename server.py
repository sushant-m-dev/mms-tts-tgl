# server.py
from concurrent import futures
import grpc
import prompt_service_pb2
import prompt_service_pb2_grpc
import torch
from transformers import VitsTokenizer, VitsModel
import scipy
import logging
import sys

#Configure logging to write log messages to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Initialize MMS TTS components
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-tgl")
model = VitsModel.from_pretrained("facebook/mms-tts-tgl")

class PromptService(prompt_service_pb2_grpc.PromptServiceServicer):
    def ProcessPrompt(self, request, context):
        try:
            text = request.prompt

            #updated_text = "So you want to name the audio as {}".format(text)


            # Convert the text prompt to audio
            inputs = tokenizer(text=request.prompt, return_tensors="pt")

            with torch.no_grad():
                output = model(**inputs).waveform

            # Save the generated audio as a WAV file
            output_file = "output.wav"
            scipy.io.wavfile.write(output_file, rate=model.config.sampling_rate, data=output.float().numpy())

            #Return success and the path to the generated audio file
            #return prompt_service_pb2.PromptResponse(success=True, message = updated_text)
            return prompt_service_pb2.PromptResponse(success=True, message="Audio generated successfully", audio_path=output_file)

        except Exception as e:
            logging.error(f"Error generating audio: {str(e)}")
            return prompt_service_pb2.PromptResponse(success=False, message="Audio generation failed")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prompt_service_pb2_grpc.add_PromptServiceServicer_to_server(PromptService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
