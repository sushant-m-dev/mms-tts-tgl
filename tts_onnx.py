import numpy as np
import os
import json
import sys
import time
import torch

from pathlib import Path
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager

# config = VitsConfig()

# config.load_json("/mnt/mydata/vits_onnx/test_server/mms-tts-tgl/eng/config.json")
# vits = Vits.init_from_config(config)

vits.load_fairseq_checkpoint(config = "/mnt/mydata/vits_onnx/test_server/mms-tts-tgl/eng/config.json", checkpoint_dir = "/mnt/mydata/vits_onnx/test_server/mms-tts-tgl/eng/G_100000.pth")


vits.export_onnx()
vits.load_onnx("coqui_mms.onnx")

text_prompt = "Hello, this is a test to determine if our model is working"

text_inputs = np.asarray(
    vits.tokenizer.text_to_ids(text_prompt, language="en"),
    dtype=np.int64,
)[None, :]

start = time.time()
audio1 = vits.inference_onnx(text_inputs)
end = time.time()
print("Inference 1 Time Taken: ", end - start, " seconds")