import numpy as np
import pyaudio
import time
import librosa
import features as ft


class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        # res = ft.compute_microphone_features(self.stream)
        # print("RES ", res
        # )
        print("STREAM", self.stream)
        self.stream.close()

        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        res = librosa.feature.mfcc(numpy_array)
        print(res)
        # print(numpy_array)

        return None, pyaudio.paContinue

    def mainloop(self):
        # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
        count = 0
        while (count < 5):
            time.sleep(2.0)
            count += 1


audio = AudioHandler()
audio.start()     # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()
