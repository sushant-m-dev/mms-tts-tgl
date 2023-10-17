import grpc
from concurrent import futures
import time
import numpy as np
from tts_pb2 import TextRequest, AudioResponse
from tts_pb2_grpc import TextToSpeechServicer, add_TextToSpeechServicer_to_server

from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig

class TextToSpeechService(TextToSpeechServicer):
    def __init__(self, vits_model):
        self.vits = vits_model

    def ConvertTextToSpeech(self, request, context):
        text_prompt = request.text

        text_inputs = np.asarray(
            self.vits.tokenizer.text_to_ids(text_prompt, language="en"),
            dtype=np.int64,
        )[None, :]

        audio_data = self.vits.inference_onnx(text_inputs)

        return AudioResponse(audio=audio_data.tobytes())

def serve():
    # Initialize the TTS model once outside the class
    config = VitsConfig()
    vits = Vits.init_from_config(config)
    vits.load_fairseq_checkpoint(config, checkpoint_dir="/mnt/mydata/ONNX_models/")
    vits.export_onnx()
    vits.load_onnx("coqui_vits.onnx")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_TextToSpeechServicer_to_server(TextToSpeechService(vits), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)  # 1 day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
