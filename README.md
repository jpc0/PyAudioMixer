PyAudioMixer
============

Advanced Realtime Software Mixer
--------------------------------

Copyright 2008, Nathan Whitehead 
Released under the LGPL

Portions Copyright 2014, Nick Vahalik (KF5ZQE)
Released under the LGPL v2.1

Portions Copyright 2020, Jean-Pierre Coetzee
Release under the LGPL v2.1

This module implements a realtime sound mixer suitable for use in
games or other audio applications.  It supports loading sounds in
uncompressed WAV format and also MP3 format.  It can mix several
sounds together during playback.  The volume and position of each
sound can be finely controlled.  Sounds are automatically resampled
and stereo converted for correct playback.  Samples can also be looped
any number of times.  Longer sounds can be streamed from a file to
save memory.  In addition, the mixer supports audio input during
playback (if supported in pyaudio with your sound card).

It has been further extended to support multiple simultaneous mixers
which can be controlled independently, frequency and DTMF generators
as well as multiple Microphone input support.

**This code is a work in progress!**

Interfaces and objects are going to be changing drastically as work 
progresses! **Use at your own risk!**

Patches welcome!

Requirements
------------

PyAudio 0.2.0 (or more recent)
http://people.csail.mit.edu/hubert/pyaudio/

NumPy 1.0 (or more recent)
http://numpy.scipy.org/

ffmpeg installed and in your path
https://ffmpeg.org/

Documentation
-------------

This README file along with the pydoc documentation in the doc/
directory are the documentation for SWMixer.


How can it possibly work in Python?
-----------------------------------

Realtime mixing of sample data is done entirely in Python using the
high performance of array operations in NumPy.  Converting between
sound formats (e.g. mono->stereo) is done using various NumPy
operations.  Resampling is done using the linear interpolation
function of NumPy.  Simultaneous playback and recording is possibly
using PyAudio.

At time of current writing, the latency and CPU utilization of 
PyAudioMixer is slightly better than Audacity running on my test
machine (a 2013 Retina MacBook Pro).


How do I use it?
----------------

Unfortunately you will need to read the code, there is a function
at the bottom of the code showing how to play a file and how to
get mic input.

Bugs and Limitations
--------------------

Always outputs in 16-bit mode.

Cannot deal with 24-bit WAV files, but CAN handle 32-bit ones
(limitation of NumPy).

Resampling can be slow for longer files.

Does not detect samplerates that differ from requested samplerates.
I.e.  if you request a rate your card cannot handle, you might get
incorrect playback rates.

Currently there is no way to limit the number of sounds mixed at once
to prevent excessive CPU usage.

No way to pan mono sounds to different positions in stereo output.

Threading behavior may not be optimal on some platforms.
