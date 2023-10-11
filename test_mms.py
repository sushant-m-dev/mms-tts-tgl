from transformers import VitsModel, AutoTokenizer
import torch
import scipy

model = VitsModel.from_pretrained("facebook/mms-tts-tgl")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tgl")

text = "interesado ka bang bumili ng pautang mula sa aming kumpanya sa 5% na rate ng interes?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform


scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
