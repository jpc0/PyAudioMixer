"""Advanced Realtime Software Mixer

This module implements an advanced realtime sound mixer suitable for
use in games or other audio applications.  It supports loading any sounds
that FFMPEG can play.  It can mix several sounds together during
playback.  The volume and position of each sound can be finely
controlled.  The mixer can use a separate thread so clients never block
during operations.  Samples can also be looped any number of times.
Looping is accurate down to a single sample, so well designed loops
will play seamlessly.  Also supports sound recording during playback.

Copyright 2008, Nathan Whitehead
Released under the LGPL

Modified by Nick Vahalik, 2014. Rewritten to allow for the creation of 
multiple discrete mixers that can be used when needing to handle
several different inputs and outputs.

Modified by Jean-Pierre Coetzee, 2020. Updated to python 3.6 and 
took development into a different direction.
"""

import subprocess
import time
import threading
import numpy
import pyaudio

__all__ = ['resample', 'interleave', 'uninterleave',
           'stereo_to_mono', 'Sound', 'MicInput', 'Mixer', 'Output', 'Clock',
           'Raw_PCM_In', 'Raw_PCM_Out']


def resample(smp, scale=1.0):
    """Resample a sound to be a different length

    Sample must be mono.  May take some time for longer sounds
    sampled at 44100 Hz.

    Keyword arguments:
    scale - scale factor for length of sound (2.0 means double length)

    """
    # f*ing cool, numpy can do this with one command
    # calculate new length of sample
    n = round(len(smp) * scale)
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    return numpy.interp(
        numpy.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        numpy.linspace(0.0, 1.0, len(smp), endpoint=False),  # known positions
        smp,  # known data points
    )


def interleave(left, right):
    """Given two separate arrays, return a new interleaved array

    This function is useful for converting separate left/right audio
    streams into one stereo audio stream.  Input arrays and returned
    array are Numpy arrays.

    See also: uninterleave()

    """
    return numpy.ravel(numpy.vstack((left, right)), order='F')


def uninterleave(data):
    """Given a stereo array, return separate left and right streams

    This function converts one array representing interleaved left and
    right audio streams into separate left and right arrays.  The return
    value is a list of length two.  Input array and output arrays are all
    Numpy arrays.

    See also: interleave()

    """
    return data.reshape(2, len(data)/2, order='FORTRAN')


def stereo_to_mono(left, right):
    """Return mono array from left and right sound stream arrays"""
    return (0.5 * left + 0.5 * right).astype(numpy.int16)


class Raw_PCM_In:
    def __init__(self, mixer):
        self.mixer = mixer
        self.done = False
        self.data = [b'']

    def get_samples(self, number_of_samples_requested):
        try:
            samples = numpy.fromstring(self.data.pop(), dtype=numpy.int16)
        except:
            samples = numpy.fromstring(b'', dtype=numpy.int16)

        if len(samples) < number_of_samples_requested:
            samples = numpy.append(samples, numpy.zeros(
                number_of_samples_requested - len(samples), numpy.int16))

        return samples

    def send_PCM(self, data):
        self.data.append(data)

    def play(self):
        self.mixer.lock.acquire()
        self.mixer.srcs.append(self)
        self.mixer.lock.release()

    def stop(self):
        self.mixer.lock.acquire()
        self.mixer.srcs.remove(self)
        self.mixer.lock.release()


class Sound:
    def __init__(self, mixer, filename, duration=-1, loop=0):
        self.mixer = mixer
        self.done = False

        if filename is None:
            assert False
        self.filename = filename

        play = ["ffmpeg", '-re',
                '-stream_loop', str(loop),
                '-i', filename,
                '-f', 's16le',
                '-ar', str(mixer.samplerate),
                '-ac', str(mixer.channels),
                'pipe:1']
        self.stream = subprocess.Popen(
            play, stdout=subprocess.PIPE)

    def set_duration(self, duration):
        self.duration = duration
        self.samples_remaining = duration * self.mixer.samplerate * self.mixer.channels

    def get_samples(self, number_of_samples_requested):
        samples = self.stream.stdout.read(
            (number_of_samples_requested * self.mixer.channels))
        samples = numpy.frombuffer(samples, dtype=numpy.int16)

        if len(samples) < number_of_samples_requested:
            self.done = True
            samples = numpy.append(samples, numpy.zeros(
                number_of_samples_requested - len(samples), numpy.int16))
            self.stream.stdout.close()
            self.done = True

        return samples

    def play(self, duration=.5, volume=1.0, offset=0, fadein=0, envelope=None):
        self.mixer.lock.acquire()
        self.mixer.srcs.append(self)
        self.mixer.lock.release()

    def stop(self):
        self.mixer.lock.acquire()
        self.mixer.srcs.remove(self)
        self.mixer.lock.release()


class MicInput:
    def __init__(self, mixer, device_id=-1, duration=-1):
        self.done = False
        self.mixer = mixer

        self.pyaudio = pa = pyaudio.PyAudio()
        self.stream = pa.open(format=pyaudio.paInt16,
                              channels=mixer.channels,
                              rate=mixer.samplerate,
                              input_device_index=device_id,
                              input=True)

    def get_samples(self, number_of_samples_requested):
        samples = self.stream.read(
            int((number_of_samples_requested) / self.mixer.channels), exception_on_overflow=False)
        samples = numpy.frombuffer(samples, dtype=numpy.int16)

        if len(samples) < number_of_samples_requested:
            self.done = True
            self.stream.close()
            # In this case we ran out of stream data
            # append zeros (don't try to be sample accurate for streams)
            samples = numpy.append(samples, numpy.zeros(
                number_of_samples_requested - len(samples), numpy.int16))

        return samples

    def unmute(self):
        self.mixer.lock.acquire()
        self.mixer.srcs.append(self)
        self.mixer.lock.release()

    def mute(self):
        self.mixer.lock.acquire()
        self.mixer.srcs.remove(self)
        self.mixer.lock.release()


class Mixer:
    def __init__(self, clock, samplerate=48000, chunksize=2**10, stereo=True):
        """Initialize mixer

        Must be called before any sounds can be played or loaded.

        Keyword arguments:
        samplerate - samplerate to use for playback (default 22050)
        chunksize - size of playback chunks
          smaller is more responsive but perhaps stutters
          larger is more buffered, less stuttery but less responsive
          Can be any size, does not need to be a power of two. (default 1024)
        stereo - whether to play back in stereo
        """
        # Checks
        assert (8000 <= samplerate <= 48000)
        assert (stereo in [True, False])

        # Mixer settings
        self.samplerate = samplerate
        self.chunksize = chunksize
        self.stereo = stereo
        if stereo:
            self.channels = 2
        else:
            self.channels = 1
        self.samplewidth = 2
        clock.add_mixer(self)

        # Variables
        self.srcs = []
        self.dests = []
        self.lock = threading.Lock()

    def tick(self, extra=None):
        """Main loop of mixer, mix and do audio IO

        Audio sources are mixed by addition and then clipped.  Too many
        loud sources will cause distortion.

        Extra is for extra sound data to mix into output

        Must be in numpy array of correct length

        """

        # Variables

        srcrmlist = []  # Sources to be removed

        # Create buffer
        buffsize = self.chunksize * self.channels
        buss = numpy.zeros(buffsize, numpy.float)

        if self.lock is None:
            return  # this can happen if main thread quit first

        self.lock.acquire()
        for sndevt in self.srcs:
            s = sndevt.get_samples(buffsize)
            if s is not None:
                buss += s
            if sndevt.done:
                srcrmlist.append(sndevt)
        if extra is not None:
            buss += extra
        buss = buss.clip(-32767.0, 32767.0)
        self.buss = buss
        for e in srcrmlist:
            self.srcs.remove(e)
        self.odata = (buss.astype(numpy.int16)).tobytes()
        for output in self.dests:
            output.play_to(self.odata)
        self.lock.release()

    def quit(self):
        """Stop all playback and terminate mixer"""
        self.lock.acquire()
        self.init = False
        self.lock.release()

    def set_chunksize(self, size=1024):
        """Set the audio chunk size for each frame of audio output

        This function is useful for setting the framerate when audio output
        is synchronized with video.
        """
        self.lock.acquire()
        self.chunksize = size
        self.lock.release()


class Output:
    '''Basic output class for speakers, can be overwritten for other output types, IE ffmpeg'''

    def __init__(self, mixer, output_device_index=-1, checks=True):
        # Variables
        self.mixer = mixer
        self.output_device_index = output_device_index

        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.mixer.channels,
            rate=self.mixer.samplerate,
            output_device_index=self.output_device_index,
            output=True)

    def start(self):
        self.mixer.lock.acquire()
        self.mixer.dests.append(self)
        self.mixer.lock.release()

    def stop(self):
        self.mixer.lock.acquire()

        self.mixer.dests.remove(self)
        self.stream.stop_stream()
        self.pyaudio.terminate()

        self.mixer.lock.release()

    def play_to(self, data):
        self.data = data
        self.stream.write(data, self.mixer.chunksize)


class Raw_PCM_Out:
    def __init__(self, mixer,):
        self.mixer = mixer
        self.data = [b'']

    def play_to(self, data):
        self.data.append(data)

    def get_PCM(self, data):
        try:
            return self.data.pop()
        except:
            return b''


class Clock:
    def __init__(self):
        self.devices = []
        clock1 = threading.Thread(target=self.run)
        clock1.start()

    def add_mixer(self, mixers):
        self.devices.append(mixers)

    def run(self):
        while True:
            for mixer in self.devices:
                mixer.tick()
            time.sleep(1/48000)

'''
Example Code

def stream1():
    mix = Mixer(clock)
    song = Sound(mix, "test/test1.wav", loop=5)
    song.play()
    mic = MicInput(mix)
    mic.unmute()
    speakers = Output(mix)
    speakers.start()

# To get device index please run audiodevicename.py


if __name__ == "__main__":
    # Clock runs in a seperate thread
    clock = Clock()
    t1 = threading.Thread(target=stream1)
    t1.start()
'''