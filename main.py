import json
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
# modules for TTS
from ibm_watson import TextToSpeechV1, SpeechToTextV1
from pyaudio import PyAudio
# modules for STT
from ibm_watson.websocket import RecognizeCallback, AudioSource
from io import BytesIO

# credentials
with open('credentials.txt') as c:
    credentials = json.load(c)
TTSauthenticator = IAMAuthenticator(credentials['TTSapi'])
text_to_speech = TextToSpeechV1(authenticator=TTSauthenticator)
text_to_speech.set_service_url(credentials['TTSurl'])
STTauthenticator = IAMAuthenticator(credentials['STTapi'])
speech_to_text = SpeechToTextV1(authenticator=STTauthenticator)
speech_to_text.set_service_url(credentials['STTurl'])

buffer = 1024
rate = 22050
width = 2
channels = 1
record_time = 3


def speak(text):
    """
    Text To Speech core. Using Watson API to get binary data and convert them into .wav
    :param text:
    :return:
    """
    p = PyAudio()

    stream = p.open(format=p.get_format_from_width(width),
                    channels=channels,
                    rate=rate,
                    output=True)

    data = text_to_speech.synthesize(
        text,
        voice='en-US_KevinV3Voice',
        accept='audio/wav'
    ).get_result().content.split(b'data', 1)[1]

    while data != b'':
        stream.write(data[:buffer + 1])
        data = data[buffer:]

    stream.stop_stream()
    stream.close()
    p.terminate()


def listen():
    """
    Speech To Text core
    :return:
    """
    p = PyAudio()

    stream = p.open(format=p.get_format_from_width(width),
                    channels=channels,
                    rate=rate,
                    input=True,
                    output=True,
                    )

    print("Listenning...")

    received = b'RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"V\x00\x00D\xac\x00\x00\x02\x00\x10\x00LIST\x1a\x00\x00\x00INFOISFT\x0e\x00\x00\x00Lavf58.29.100\x00data'

    for i in range(0, int(rate / buffer * record_time)):
        received += stream.read(buffer)

    print("Processing...")

    received_data = BytesIO(received)

    class MyRecognizeCallback(RecognizeCallback):
        def __init__(self):
            RecognizeCallback.__init__(self)
            self.result = 'Couldn\'t hear what you said. Please try again later'

        def on_data(self, data):
            self.result = data['results'][0]['alternatives'][0]['transcript']

        def on_error(self, error):
            self.result = 'Error received: {}'.format(error)

        def on_inactivity_timeout(self, error):
            self.result = 'Inactivity timeout: {}'.format(error)

    my_recognize_callback = MyRecognizeCallback()

    audio_source = AudioSource(received_data)
    speech_to_text.recognize_using_websocket(
        audio=audio_source,
        content_type='audio/wav',
        recognize_callback=my_recognize_callback,
        model='en-US_BroadbandModel'
    )

    received_data.close()
    stream.stop_stream()
    stream.close()
    p.terminate()

    return my_recognize_callback.result
