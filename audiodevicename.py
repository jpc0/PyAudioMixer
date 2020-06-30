import pyaudio

test = pyaudio.PyAudio()

audioindevice = {}
audiooutdevice = {}

for i in range(test.get_device_count()):
    if (test.get_device_info_by_index(i)['maxInputChannels'] > 0):
        audioindevice[i] = test.get_device_info_by_index(i)['name']
    else:
        pass

for i in range(test.get_device_count()):    
    if (test.get_device_info_by_index(i)['maxOutputChannels'] > 0):
        audiooutdevice[i] = test.get_device_info_by_index(i)['name']
    else:
        pass


print("Audio input devicess: " + str(audioindevice) + "\n")
print("Audio output devices: " + str(audiooutdevice) + "\n")
'''
for i in range(test.get_device_count()):
    print(test.get_device_info_by_index(i)['name'])
    print("\n")
'''