import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import os
import sys
import random
from scipy.signal import butter, lfilter
from scipy.optimize import minimize
from copy import deepcopy
def startWith(a):
    for x in a:
        if x[:6] == "pid0_0":
            return x[14:-1]
import os
iter = 0
# levenshteinDistance
def loss1(predicted, target):
    global iter
    iter += 1
    if iter % 10 == 0:
        print('Loop', iter)
        print('predicted:', predicted)
        print('target:', target)
    s1, s2 = predicted, target

    if len(s1) > len(s2):
        s1, s2 = s2, s1
        
    distances = range(len(s1) + 1) 
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    if iter % 10 == 0: print('Loss:', distances[-1])
    return distances[-1]

# ctcloss
def loss2(predicted, target):
    batch_size = 1


def runtest():
    filepath = "\"/home/csa_james/Downloads/CurrentProjects/bg/on_off.wav\""
    filepath = input("File path: ")
    target = input("Target: ")
    stream = os.popen("python3 funasr_wss_client.py --host \"127.0.0.1\" --port 10095 --mode offline --audio_in "+filepath)
    output = stream.read()
    output = output.split("\n")
    predicted = startWith(output)

    print("Recognized As:" ,predicted)
    print('Loss:', loss1(target, predicted))

# soundarray is an sound array, returns predicted string
def runModel(soundarray, name):
    # create new file for the new sound array
    cur_path = os.getcwd()
    name = os.path.join(cur_path, "rec", name + '.wav')
    wav.write(name, 16000, np.array(np.clip(np.round(soundarray), -2**15, 2**15-1), dtype=np.int16))

    # return predicted string (or logits in the future)
    stream = os.popen("python3 funasr_wss_client.py --host \"127.0.0.1\" --port 10095 --mode offline --audio_in "+name)
    output = stream.read()
    output = output.split("\n")
    predicted = startWith(output)
    return predicted

# return a list of audio arrays
def get_bg_noise_audio_sets():
    directory_path = '/home/csa_james/Downloads/CurrentProjects/PitchShift/30samplesII'
    file_names = os.listdir(directory_path)
    ans = []
    for name in file_names:
        name = directory_path +"/"+ name
        fs, audio = wav.read(name)
        ans.append(audio)
    for i ,audio in enumerate(ans):
        if audio.ndim > 1 and audio.shape[1] == 2:  # Check if audio is stereo
            audio = np.mean(audio, axis=1)
            ans[i] = audio
       
    return ans


def func1(filepath, target):
    def constraint(x):
                return sum(x) - 0.5
   
    options = {'eps': 1e-5}
    bg_sets = get_bg_noise_audio_sets()
    bounds =[(0, 0.5)]*len(bg_sets)
    fs, audio = wav.read(filepath)

    # scaled_bg = 0.5 * bg_sets[0][:len(audio)]
    # audio += np.array(scaled_bg, dtype=np.int16)

    weights = [0.5/len(bg_sets)] * len(bg_sets)

    
    
    # print(loss1(predicted, target))
    def getloss1(weights):
        global iter

        cur_path = os.getcwd()
        name = os.path.join(cur_path, "rec", 'reccord.txt')
        with open(name, 'a') as file:
            file.write(str(iter)+' ' + str(weights)+'\n\n')
        audio_w_bg = deepcopy(audio)

        for i, w in enumerate(weights):
            # Ensure the background noise is mono
            scaled_bg = w * bg_sets[i]
            scaled_bg = scaled_bg[:len(audio_w_bg)]
            # Ensure the scaled background is in the correct shape and type before adding
            audio_w_bg += np.array(scaled_bg, dtype=np.int16)

        predicted = runModel(audio_w_bg, "test1." + str(iter))
        return loss1(predicted, target)
    weights = """0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00050403 0.00050403 0.00050403 0.00050403 0.00050403
        0.00050403 0.00051403"""
    weights = list(map(float, weights.split()))

    # ans = minimize(getloss1, weights, method= 'SLSQP', constraints={'type': 'eq', 'fun': constraint}, bounds=bounds, options=options)
    return getloss1(weights)

# print(minimize(self.get_only_loss, x, method= 'SLSQP', constraints={'type': 'eq', 'fun': constraint}, bounds=bounds, options=options))


ans = func1("/home/csa_james/Downloads/CurrentProjects/bg/on_off.wav", "off")
print(ans)
'''





'''