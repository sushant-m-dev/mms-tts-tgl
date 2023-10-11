import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import scipy
import sys,logging

# Configure logging to write log messages to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  # make deterministic

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
logging.info(waveform)

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=waveform)

