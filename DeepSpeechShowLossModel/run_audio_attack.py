import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import os
import sys
sys.path.append("DeepSpeech")
import random
from scipy.signal import butter, lfilter
from scipy.optimize import minimize
from copy import deepcopy
###########################################################################
# TWO WAYS TO IMPROVE THE CODE
#       1. READ THE PAPER AND TRY TO MANIPULATE THE PITCH/SPEECHRATE
#       2. USE SPSA ALGORITHM


###########################################################################

###########################################################################
# This section of code is credited to:
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
generation_tmp = os.path.exists
os.path.exists = lambda x: True

class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]

class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v

tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = generation_tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

from util.text import ctc_label_dense_to_sparse
from tf_logits import compute_mfcc, get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

###########################################################################

def db(audio):
    # if the audio has more than 1 dimension
    if len(audio.shape) > 1:
        maxx = np.max(np.abs(audio), axis=1)
        return 20 * np.log10(maxx) if np.any(maxx != 0) else np.array([0])
    maxx = np.max(np.abs(audio))
    return 20 * np.log10(maxx) if maxx != 0 else np.array([0])

def load_wav(input_wav_file):
    # Load the inputs that we're given
    fs, audio = wav.read(input_wav_file)
    assert fs == 16000
    return audio

def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))
    print('output dB', db(audio))

def save_wav_rec(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))
    print('output dB', db(audio))
def levenshteinDistance(s1, s2): 
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
    return distances[-1]

# Cut off low frequency sound to not make it sound obvious
def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)

# Initialize the first population
def get_new_pop(elite_pop, elite_pop_scores, pop_size):
    scores_logits = np.exp(elite_pop_scores - elite_pop_scores.max()) 
    elite_pop_probs = scores_logits / scores_logits.sum()
    cand1 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    cand2 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    mask = np.random.rand(pop_size, elite_pop.shape[1]) < 0.5 
    next_pop = mask * cand1 + (1 - mask) * cand2
    return next_pop

###########################################################################
# The add noise to the file part
# It's the only modification to make the OG to the goal
# TODO: how to add background noise?
# Do we have population?
def mutate_pop(pop, mutation_p, noise_stdev, elite_pop):
    # unpack pop elements and pass them as seperate arguments
    # print(*[1, 2, 3]) -> 1, 2, 3
    # noise_stdev represents the standard deviation in the Normal distribution
    noise = np.random.randn(*pop.shape) * noise_stdev
    noise = highpass_filter(noise)

    # create a mask that has the same shape as the elite population - only the ones that smaller than the threshold - the mutation_p will be selected
    # shape[0] returns the number of individuals, shape[1] returns the number of genes
    # why don't you just use pop.shape[1]? just wanna make sure that you create a mask for the elite pop
    mask = np.random.rand(pop.shape[0], elite_pop.shape[1]) < mutation_p 
    new_pop = pop + noise * mask
    return new_pop
    

    # Create a new variable called bg_noise
    # mask will change according to the standard deviation (line 250)

###########################################################################
class Genetic():
    
    def __init__(self, bg_noise_files,input_wave_file, output_wave_file, target_phrase):
        self.pop_size = 100
        self.elite_size = 10
        self.mutation_p = 0.005
        self.noise_stdev = 40
        self.noise_threshold = 1
        self.mu = 0.9
        self.alpha = 0.001
        self.max_iters = 100
        self.num_points_estimate = 100
        self.delta_for_gradient = 100
        self.delta_for_perturbation = 1e3
        self.bg_noise_audio_sets = []
        self.counter = 0

        ## PROCESSING THE AUDIO FILES
        self.bg_scale = 0.1

        self.input_audio = load_wav(input_wave_file).astype(np.float32)
        for bg_noise_wave_file in os.listdir(bg_noise_files):
            bg_noise_wave_file = bg_noise_files +'/' + bg_noise_wave_file
            # load background noise
            bg_noise_audio = load_wav(bg_noise_wave_file).astype(np.float32)

            # clip background file
            bg_noise_audio = bg_noise_audio[:len(self.input_audio)]
            self.bg_noise_audio_sets.append(bg_noise_audio)
        # add background noise to input audio
        self.input_audio_w_bg = self.input_audio 


        # input audio from 1D array to 2D array
        self.pop = np.expand_dims(self.input_audio_w_bg, axis=0)
        # multiply into 100 candidates: (1, 32) x (100, 1) -> (100, 32)
        self.pop = np.tile(self.pop, (self.pop_size, 1))
        self.output_wave_file = output_wave_file
        self.target_phrase = target_phrase
        self.funcs = self.setup_graph(self.pop, np.array([toks.index(x) for x in target_phrase]))

    def setup_graph(self, input_audio_batch, target_phrase): 
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320 
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)
        
        pass_in = np.clip(input_audio_batch, -2**15, 2**15-1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            
            inputs = tf.placeholder(tf.float32, shape=pass_in.shape, name='a')
            len_batch = tf.placeholder(tf.float32, name='b')
            arg2_logits = tf.placeholder(tf.int32, shape=logits_arg2.shape, name='c')
            arg1_dense = tf.placeholder(tf.float32, shape=dense_arg1.shape, name='d')
            arg2_dense = tf.placeholder(tf.int32, shape=dense_arg2.shape, name='e')
            len_seq = tf.placeholder(tf.int32, shape=seq_len.shape, name='f')
            
            logits = get_logits(inputs, arg2_logits)
            target = ctc_label_dense_to_sparse(arg1_dense, arg2_dense, len_batch)
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=len_seq)
            decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg2_logits, merge_repeated=True)
            
            sess = tf.Session()
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, "models/session_dump")
            
        func1 = lambda a, b, c, d, e, f: sess.run(ctcloss, 
            feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d, arg2_dense: e, len_seq: f})
        func2 = lambda a, b, c, d, e, f: sess.run([ctcloss, decoded], 
            feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d, arg2_dense: e, len_seq: f})
        return (func1, func2)

    def getctcloss(self, input_audio_batch, target_phrase, decode=False):
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320 
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)
        
        pass_in = np.clip(input_audio_batch, -2**15, 2**15-1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)
        if decode:
            return self.funcs[1](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
        else:
            return self.funcs[0](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
        
    def get_fitness_score(self, input_audio_batch, target_phrase, input_audio, classify=False):
        target_enc = np.array([toks.index(x) for x in target_phrase]) 
        if classify:
            ctcloss, decoded = self.getctcloss(input_audio_batch, target_enc, decode=True)
            all_text = "".join([toks[x] for x in decoded[0].values]) 
            index = len(all_text) // input_audio_batch.shape[0] 
            final_text = all_text[:index]
        else:
            ctcloss = self.getctcloss(input_audio_batch, target_enc)
        score = -ctcloss
        if classify:
            return (score, final_text) 
        return score, -ctcloss

    
   
    # used to be run    
    def run(self, log=None):
        def constraint(x):
            return sum(x) - 0.5
        max_fitness_score = float('-inf') 
        dist = float('inf')
        best_text = ''
        itr = 1
        len_bg = len(self.bg_noise_audio_sets)
        while itr < 2:
            if log is not None:
                log.write('target phrase: ' + self.target_phrase + '\n')
                log.write('itr, corr, lev dist \n')
            
            bounds =[(0, 0.5)]*len_bg
            options = {'eps': 1e-5}

            
            x =[0/len_bg]*len_bg
            
            # self.input_audio_w_bg = self.input_audio + (self.bg_noise_audio_sets[0]*a1 + self.bg_noise_audio_sets[1]*a2 + self.bg_noise_audio_sets[2]*a3 + self.bg_noise_audio_sets[3]*a4 + self.bg_noise_audio_sets[4]*a5 + self.bg_noise_audio_sets[5]*a6 + self.bg_noise_audio_sets[6]*a7 + self.bg_noise_audio_sets[7]*a8 + self.bg_noise_audio_sets[8]*a9 + self.bg_noise_audio_sets[9]*a10 + self.bg_noise_audio_sets[10]*a11 + self.bg_noise_audio_sets[11]*a12 + self.bg_noise_audio_sets[12]*a13 )
            self.input_audio_w_bg = self.input_audio
            for i in range(len_bg):
                self.input_audio_w_bg += self.bg_noise_audio_sets[i]*x[i]
            self.pop = np.expand_dims(self.input_audio_w_bg, axis=0)
            self.pop = np.tile(self.pop, (self.pop_size, 1))

            pop_scores, ctc = self.get_fitness_score(self.pop, self.target_phrase, self.input_audio_w_bg)
            elite_ind = np.argsort(pop_scores)[-self.elite_size:]
            elite_pop, elite_pop_scores, elite_ctc = self.pop[elite_ind], pop_scores[elite_ind], ctc[elite_ind]
            
            

            if itr % 1 == 0:
                print('**************************** RESULT ****************************'.format(itr))
                print('Current loss: {}'.format(-elite_ctc[-1]))
                # choose the best candidate, aka elite_pop[-1], because the pop is sorted from low to high
                save_wav(elite_pop[-1], self.output_wave_file)
                best_pop = np.tile(np.expand_dims(elite_pop[-1], axis=0), (100, 1))
                _, best_text = self.get_fitness_score(best_pop, self.target_phrase, self.input_audio_w_bg, classify=True)
                
                dist = levenshteinDistance(best_text, self.target_phrase)
                corr = "{0:.4f}".format(np.corrcoef([self.input_audio_w_bg, elite_pop[-1]])[0][1])
                print('Audio similarity to input: {}'.format(corr))
                print('Edit distance to target: {}'.format(dist))
                print('Currently decoded as: {}'.format(best_text))
                if log is not None:
                    log.write(str(itr) + ", " + corr + ", " + str(dist) + "\n")
            


            ### need to change the pop to go to the next iteration

            itr += 1


            
        # The return value is not important
        # the output file is
        return itr < self.max_iters
    


bg_noise_files = sys.argv[1]
inp_wav_file = sys.argv[2]
target = sys.argv[3].lower()
log_file = inp_wav_file[:-4] + '_log.txt'
out_wav_file = inp_wav_file[:-4] + '_adv.wav'

print('target phrase:', target)
print('source file:', inp_wav_file)

g = Genetic(bg_noise_files, inp_wav_file, out_wav_file, target)
with open(log_file, 'w') as log:

    success = g.run(log=log)

if success:
	print('Succes! Wav file stored as', out_wav_file)
else:
	print('Not totally a success! Consider running for more iterations. Intermediate output stored as', out_wav_file)

