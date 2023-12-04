import socket
import soundfile as sf
import sounddevice as sd
import numpy as np
import librosa
import threading
from tensorflow import keras

# Turn On(1), Off(0) ANC Function
ANC_ON = 1

# Create a Bluetooth RFCOMM socket
client_socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
client_socket.connect(("80:38:FB:4D:A7:73", 10))

# Audio Sampling, STFT Properties
sample_rate = 11025
blocksize = 2304
n_fft = 144

# Load LSTM Model and Reset Values
model = keras.models.load_model('anc_model_2/')
wav_weight_l = np.zeros((73,33))
wav_weight_r = np.zeros((73,33))
noise_audio_sample_l = np.zeros(blocksize)
noise_audio_sample_r = np.zeros(blocksize)
noise_audio_input_l = np.zeros(blocksize)
noise_audio_input_r = np.zeros(blocksize)

# Calculate Predicted Noise Weight Map
def noise_weight():
    global wav_weight_l
    global wav_weight_r
    pred_mag_ratio_l = 1
    pred_mag_ratio_r = 1
    input_mag_ratio_l = 1
    input_mag_ratio_r = 1

    threading.Timer(5, noise_weight).start()

    if noise_audio_sample_l.any() and noise_audio_sample_r.any() and ANC_ON:
        outdata_stft_l = librosa.stft(noise_audio_sample_l,n_fft=n_fft)
        outdata_stft_r = librosa.stft(noise_audio_sample_r,n_fft=n_fft)

        mag_l_origin, _ = librosa.magphase(outdata_stft_l)
        mag_r_origin, _ = librosa.magphase(outdata_stft_r)

        mag_l = mag_l_origin.T
        mag_l = mag_l.reshape(mag_l.shape[0], 1, mag_l.shape[1])
        pred_mag_l = model.predict(mag_l)
        pred_mag_l = pred_mag_l.T
        pred_mag_l = pred_mag_l.reshape(pred_mag_l.shape[0], pred_mag_l.shape[2])
        pred_mag_ratio_l = np.sqrt(np.mean(pred_mag_l ** 2))

        mag_r = mag_r_origin.T
        mag_r = mag_r.reshape(mag_r.shape[0], 1, mag_r.shape[1])
        pred_mag_r = model.predict(mag_r)
        pred_mag_r = pred_mag_r.T
        pred_mag_r = pred_mag_r.reshape(pred_mag_r.shape[0], pred_mag_r.shape[2])
        pred_mag_ratio_r = np.sqrt(np.mean(pred_mag_r ** 2))

        print("외부소음 수집됨")
        print(pred_mag_ratio_l, pred_mag_ratio_r)
    else:
        pass

    if noise_audio_input_l.any() and noise_audio_input_r.any() and ANC_ON:
        outdata_stft_l = librosa.stft(noise_audio_input_l,n_fft=n_fft)
        outdata_stft_r = librosa.stft(noise_audio_input_r,n_fft=n_fft)

        mag_l, _ = librosa.magphase(outdata_stft_l)
        mag_r, _ = librosa.magphase(outdata_stft_r)

        input_mag_ratio_l = np.sqrt(np.mean(mag_l ** 2))
        input_mag_ratio_r = np.sqrt(np.mean(mag_r ** 2))
        print("내부소음 수집됨")
        print(input_mag_ratio_l, input_mag_ratio_r)
    else:
        pass
    
    if ANC_ON:
        wav_weight_l = pred_mag_l * 4 * (input_mag_ratio_l / pred_mag_ratio_l)
        wav_weight_r = pred_mag_r * 4 * (input_mag_ratio_r / pred_mag_ratio_r)
    print("")

# Apply Weight Map and sound out
def callback_out(outdata, frames, time, status):
    buf = b''
    length = len(outdata)*8
    #2304
    step = length
    global n_fft
    global noise_audio_sample_l
    global noise_audio_sample_r
    global noise_audio_input_l
    global noise_audio_input_r

    try:
        while True:
            data = client_socket.recv(step)
            buf += data
            if len(buf) == length:
                break
            elif len(buf) < length:
                step = length - len(buf)
    except Exception as e:
        print(e)
    
    outdata_bit = np.frombuffer(buf, dtype=np.float32)
    length_shape = int(length/8)

    outdata_l = np.zeros(length_shape)
    outdata_r = np.zeros(length_shape)
    
    # outdata_bit looks like
    # outdata_l[0], outdata_r[0], outdata_l[1], outdata_r[1], ...
    # So, Calculate index like this way
    for i in range(int(length_shape * 2)):
        if i % 2 == 0:
            outdata_l[int(i/2)] = outdata_bit[i:i+1]
        else:
            outdata_r[int((i-1)/2)] = outdata_bit[i:i+1]
    
    noise_audio_sample_l = outdata_l
    noise_audio_sample_r = outdata_r
    
    outdata_l = noise_audio_input_l
    outdata_r = noise_audio_input_r

    outdata_stft_l = librosa.stft(outdata_l,n_fft=144)
    outdata_stft_r = librosa.stft(outdata_r,n_fft=144)

    mag_l, phase_l = librosa.magphase(outdata_stft_l)
    mag_r, phase_r = librosa.magphase(outdata_stft_r)

    if wav_weight_l.any() and wav_weight_r.any() and ANC_ON:
        pred_l = wav_weight_l*(-1) * phase_l
        pred_r = wav_weight_r*(-1) * phase_r
        print("WORKED")
    else:
        pred_l = outdata_stft_l
        pred_r = outdata_stft_r
        print("NO")
    

    outdata_istft_l = librosa.istft(pred_l,n_fft=n_fft)
    outdata_istft_r = librosa.istft(pred_r,n_fft=n_fft)
    
    if ANC_ON:
        outdata[:] = np.stack((outdata_istft_l, outdata_istft_r), axis=1)
    else:
        outdata[:] = np.stack((noise_audio_sample_l, noise_audio_sample_r), axis=1)
        #pass

def callback_in(indata, frames, time, status):
    global noise_audio_input_l
    global noise_audio_input_r

    print(indata.shape)

    noise_audio_input_l = indata[:,0]
    noise_audio_input_r = indata[:,1]

# These two functions are written because of multi-threading
def speaker():
    stream = sd.OutputStream(callback=callback_out, blocksize=blocksize, channels=2, dtype=np.float32, samplerate=sample_rate)
    stream.start() 
    sd.sleep(1000 * 30) # 스트림 유지

def mic():
    stream = sd.InputStream(callback=callback_in, blocksize=blocksize, channels=2, dtype=np.float32, samplerate=sample_rate)
    stream.start()
    sd.sleep(1000 * 30)# 스트림 유지
try:
    thread_mic = threading.Thread(target=mic, args=())
    thread_speaker = threading.Thread(target=speaker, args=())
    thread_noise = threading.Thread(target=noise_weight, args=())
    thread_mic.start()
    thread_speaker.start()
    thread_noise.start()
    
    
    print(f"Connected. Press Ctrl+C to exit.")
    sd.sleep(1000 * 300)  # 스트림 유지
except KeyboardInterrupt:
    pass
    # 연결 해제
    client_socket.close()
    
