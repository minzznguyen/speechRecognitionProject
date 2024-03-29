import torch
import numpy as np
from scipy.io import wavfile
from torch_pitch_shift import *
from pydub import AudioSegment
import sys

def pitchShift(name, octaves):
    # Load the mono audio
    mono_audio = AudioSegment.from_file(name, format="wav")

    # Duplicate the mono audio to create stereo audio
    stereo_audio = mono_audio.set_channels(2)

    # Export the stereo audio to a new file
    stereo_audio.export(name, format="wav")

    # read an audio file
    SAMPLE_RATE, sample = wavfile.read(name)

    # convert to tensor of shape (batch_size, channels, samples)
    dtype = sample.dtype
    sample = torch.tensor(
        [np.swapaxes(sample, 0, 1)],  # (samples, channels) --> (channels, samples)
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # pitch down by 12 semitones
    down = pitch_shift(sample, octaves, SAMPLE_RATE)
    assert down.shape == sample.shape
    wavfile.write(
        f"./{name[:-4]}_p{octaves}.wav",
        SAMPLE_RATE,
        np.swapaxes(down.cpu()[0].numpy(), 0, 1).astype(dtype),
    )

bg_noise_files = sys.argv[1]
temp = []
with open(bg_noise_files, 'r') as f:
    for x in f:
        temp.append(x)
        
new_files = []

for bg_noise_wave_file in temp:
    # load background noise
    l_octaves = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    for x in l_octaves:
        pitchShiftl(bg_noise_wave_fie[:-1], x)


        # store the newly created filename into a string to add later
        new_files.append(f"{bg_noise_wave_file[:-5]}_p{x}.wav\n")

with open("bg_noises_out.txt", 'a') as f:
    for x in new_files:
        f.write(x)

