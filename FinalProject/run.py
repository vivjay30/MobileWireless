import time

import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

from network import Demucs, center_trim

t0 = time.time()
SR = 22050
MODEL_PATH = "vctk_real_recording_22050_multiplevoices_betterbatch_20.pt"
INPUT_FILE = "Samples/test2.wav"

device = torch.device("cpu")
model = Demucs(n_audio_channels=2)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.train = False
model.to(device)

print(time.time() - t0)

mixed_data = librosa.core.load(INPUT_FILE, mono=False, sr=SR)[0]
data = torch.tensor(mixed_data).to(device).unsqueeze(0)

print(time.time() - t0)
# Normalize input
data = (data * 2**15).round() / 2**15
ref = data.mean(1)  # Average across the n microphones
means = ref.mean(1).unsqueeze(1).unsqueeze(2)
stds = ref.std(1).unsqueeze(1).unsqueeze(2)
data_transformed = (data - means) / stds

# Run through the model
valid_length = model.valid_length(data_transformed.shape[-1])
delta = valid_length - data_transformed.shape[-1]
padded = F.pad(data_transformed, (delta // 2, delta - delta // 2))


output_signal = model(padded)

output_signal = center_trim(output_signal, data_transformed)

output_signal = output_signal * stds.unsqueeze(3) + means.unsqueeze(3)
output_voices = output_signal[:, 0]  # batch x n_mics x n_samples

output_np = output_voices.detach().cpu().numpy()[0]
energy = librosa.feature.rms(output_np).mean()

print(time.time() - t0)


sf.write("output.wav", output_np[0], SR)
