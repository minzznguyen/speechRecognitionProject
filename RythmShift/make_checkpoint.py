import librosa
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import sys

# Load the audio file
def rythm_shift(audio_path, new_tempo):

    audio, sr = librosa.load(audio_path[:-1], sr=None)
    SAMPLE_RATE, sample = wavfile.read(audio_path[:-1])
    print('Original audio shape, sr with wavfile:', sample.shape, SAMPLE_RATE)
    print('original audio shape:', audio.shape)
    print('original sample rate:', sr)

    # Change the rhythm of the audio by applying time stretching
    new_audio = librosa.effects.time_stretch(audio, rate=new_tempo)

    # Adjust the length of the modified audio
    if new_tempo < 1:
        # Clip the audio to its original length
        original_length = len(audio)
        new_audio = new_audio[:original_length]
    else:
        # Repeat the audio to match the length of the original one
        original_length = len(audio)
        num_repeats = original_length // len(new_audio) 
        remainder = original_length % len(new_audio)
        repeated_audio = np.tile(new_audio, num_repeats)
        new_audio = np.concatenate((repeated_audio, new_audio[:remainder]))

    print("New audio shape:", new_audio.shape)

    # Save the modified audio to a new file with the original sample rate
    sf.write(
        f"{audio_path[:-5]}_x{new_tempo}.wav",
        new_audio,
        SAMPLE_RATE,
    )

    SAMPLE_RATE, sample = wavfile.read(f"{audio_path[:-5]}_x{new_tempo}.wav")
    print('New audio shape, sr with wavfile:', sample.shape, SAMPLE_RATE)



bg_noise_files = sys.argv[1]
temp = []
with open(bg_noise_files, 'r') as f:
    for x in f:
        temp.append(x)
new_files = []
print(temp)
for bg_noise_wave_file in temp:
    print(bg_noise_wave_file)

    # load background noise
    tempos = [0.3, 0.8, 0.5, 0.6, 0.7, 1.1, 1.5, 2, 2.5, 3, 1.75]
    for x in tempos:
        rythm_shift(bg_noise_wave_file, x)


        # store the newly created filename into a string to add later
        new_files.append(f"{bg_noise_wave_file[:-5]}_x{x}.wav\n")

print(new_files)
with open("bg_noises_out.txt", 'a') as f:
    for x in new_files:
        f.write(x)



