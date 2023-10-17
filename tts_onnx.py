import time
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import torch

# torch.set_num_threads(1)

path = "/home/mllopart/PycharmProjects/ONNX/venv/lib/python3.10/site-packages/TTS/.models.json"

model_manager = ModelManager(path)

model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/vits")

syn = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
)

text1 = "The field of space exploration has continually fascinated humanity, igniting the collective imagination and driving scientific and technological advancement. From the first successful launch of Sputnik 1 by the USSR in 1957, it became clear that space was a new frontier, ripe for exploration. Space exploration has offered us a unique vantage point to better understand our universe, revealing startling and wondrous phenomena like black holes, nebulae, and countless galaxies far beyond our own. It has also allowed us to study our home planet in ways that would have been impossible from the ground, enhancing our understanding of Earth's atmosphere, weather systems, and the impact of human activity on the global environment."

start_time = time.time()
outputs1 = syn.tts(text1)
end_time = time.time()
print(f"Time taken for inference 1: {end_time - start_time} seconds")
syn.save_wav(outputs1, "normal_1.wav")