from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import TextToSpeechV1
import pyaudio
import json

# credentials
with open('credentials.txt') as c:
    credentials = json.load(c)
authenticator = IAMAuthenticator(credentials['api'])
text_to_speech = TextToSpeechV1(authenticator=authenticator)
text_to_speech.set_service_url(credentials['url'])


def speak(text):
    """
    Text To Speech core. Using Watson API to get binary data and convert them into .wav
    :param text:
    :return:
    """
    buffer = 1024
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=22050,
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
