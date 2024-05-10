from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
import audioop
import wave
import json
import os

app = Flask(__name__)

model = Model("models/vosk-model-fr-0.22")
sample_rate = 16000

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'id' not in request.json:
        return jsonify({ "status": 400, 'error': 'No audio data'}), 400
    
    audio_id = request.json['id']
    filepath = f"audio/wav/{audio_id}.wav"

    if not os.path.exists(filepath):
        return jsonify({ "status": 404, 'error': 'Audio file not found'}), 404

    wf = wave.open(filepath, 'rb')

    n_channels = wf.getnchannels()

    # If not mono, convert it to mono
    if n_channels != 1:
        mono_filename = f'{filepath}.monofile.wav'
        mono = wave.open(mono_filename, 'wb')
        mono.setparams(wf.getparams())
        mono.setnchannels(1)
        mono.writeframes(audioop.tomono(wf.readframes(float('inf')), wf.getsampwidth(), 1, 1))
        mono.close()

        wf = wave.open(mono_filename, 'rb')

        os.remove(mono_filename)

    rec = KaldiRecognizer(model, wf.getframerate())

    transcription = ''
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            transcription += str(eval(rec.Result())['text'])+' '

    transcription += str(eval(rec.FinalResult())['text'])

    return jsonify({'text': transcription}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3800)
