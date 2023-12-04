import socket
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

# Properties of Noise
sample_rate = 11025

def callback(indata, frames, time, status):
    global client_socket
    client_socket.send(indata)

# Create a Bluetooth RFCOMM socket
server_socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
server_socket.bind(("80:38:FB:4D:A7:73", 10))
server_socket.listen(1)

print("Waiting for a connection...")
client_socket, client_address = server_socket.accept()
print("Connected to:", client_address)

# Send audio data in chunks

try:
    stream = sd.InputStream(callback=callback, blocksize=1152, channels=2, dtype=np.float32, samplerate=sample_rate)
    stream.start()

    # Send data for 30 seconds
    sd.sleep(1000*300)

except KeyboardInterrupt:
    print("ERROR")

client_socket.close()
server_socket.close()
