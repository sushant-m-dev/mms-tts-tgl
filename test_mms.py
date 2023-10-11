from transformers import VitsModel, AutoTokenizer
import torch
import scipy

model = VitsModel.from_pretrained("facebook/mms-tts-en")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-en")

text = "are you interested in buying loan from our company at 5 percent interest rate?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform


scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
