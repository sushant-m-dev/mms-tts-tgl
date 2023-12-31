import torch , numpy
from transformers import VitsTokenizer, VitsModel, set_seed
import scipy,time
import sys,logging

# Configure logging to write log messages to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-tgl")
model = VitsModel.from_pretrained("facebook/mms-tts-tgl")

inputs = tokenizer(text="Kumusta , gusto mo bang marinig ang tungkol sa aming bagong pag-aalok ng pautang na higit pa sa mga rate ng merkado", return_tensors="pt")

#set_seed(555)  # make deterministic

with torch.no_grad():
   start_time = time.time()
   output= model(**inputs).waveform
   end_time = time.time()

# waveform = outputs.waveform[0]
logging.info("Generated output in {} second".format(end_time - start_time))

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.float().numpy().T)


