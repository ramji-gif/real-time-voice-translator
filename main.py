import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import io
import os
from pydub import AudioSegment
import tempfile

app = FastAPI()
translator = Translator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Voice Translator backend is running."}

# Language map with TTS and speech-to-text languages
language_map = {
    "Hindi": ("hi-IN", "hi", "hi"),
    "English": ("en-IN", "en", "en"),
    "Tamil": ("ta-IN", "ta", "ta"),
    "Telugu": ("te-IN", "te", "te"),
    "Bengali": ("bn-IN", "bn", "bn"),
    "Urdu": ("ur-IN", "ur", "ur"),
    "Marathi": ("mr-IN", "mr", "mr"),
    "Gujarati": ("gu-IN", "gu", "gu"),
    "Kannada": ("kn-IN", "kn", "kn"),
    "Malayalam": ("ml-IN", "ml", "ml"),
    "Punjabi": ("pa-IN", "pa", "pa"),
    "Assamese": ("as-IN", "hi", "as"),
    "Odia": ("or-IN", "hi", "or"),
    "Bhojpuri": ("hi-IN", "hi", "bho"),
    "Maithili": ("hi-IN", "hi", "mai"),
    "Chhattisgarhi": ("hi-IN", "hi", "hne"),
    "Rajasthani": ("hi-IN", "hi", "raj"),
    "Konkani": ("hi-IN", "hi", "kok"),
    "Dogri": ("hi-IN", "hi", "doi"),
    "Kashmiri": ("hi-IN", "hi", "ks"),
    "Santhali": ("hi-IN", "hi", "sat"),
    "Sindhi": ("hi-IN", "hi", "sd"),
    "Manipuri": ("hi-IN", "hi", "mni"),
    "Bodo": ("hi-IN", "hi", "brx"),
    "Sanskrit": ("sa-IN", "hi", "sa")
}

# Store device WebSocket connections (mapped by device_id)
connected_devices = {}

@app.websocket("/ws/{src}/{tgt}/{device_id}")
async def translate_ws(websocket: WebSocket, src: str, tgt: str, device_id: str):
    print(f"🔌 WebSocket connection opened for {device_id} - {src} → {tgt}")
    await websocket.accept()
    recognizer = sr.Recognizer()

    src_locale, src_tts_lang, src_code = language_map.get(src, ("hi-IN", "hi-IN", "hi"))
    _, tgt_tts_lang, tgt_code = language_map.get(tgt, ("hi-IN", "hi-IN", "hi"))

    # Store the device connection
    connected_devices[device_id] = websocket

    try:
        while True:
            # Wait for audio input from this device
            audio_chunk = await websocket.receive_bytes()
            print(f"📥 Received audio blob of size {len(audio_chunk)} bytes")

            # Create a temporary file from the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as webm_file:
                webm_file.write(audio_chunk)
                webm_path = webm_file.name

            # Convert webm to wav for speech recognition
            wav_path = webm_path.replace(".webm", ".wav")
            try:
                AudioSegment.from_file(webm_path).export(wav_path, format="wav")
                print("✅ Converted webm to wav")
            except Exception as e:
                await websocket.send_text(f"Audio conversion failed: {str(e)}")
                os.remove(webm_path)
                continue

            # Recognize speech from the audio file (speech-to-text)
            try:
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=src_locale)
                print(f"🗣️ Recognized: {text}")
            except Exception as e:
                await websocket.send_text(f"STT failed: {str(e)}")
                os.remove(webm_path)
                os.remove(wav_path)
                continue

            # Translate the recognized text
            try:
                translated = translator.translate(text, src=src_code, dest=tgt_code).text
                print(f"🌐 Translated: {translated}")
            except Exception as e:
                await websocket.send_text(f"Translation failed: {str(e)}")
                continue

            # Generate TTS (Text-to-Speech) from the translated text
            try:
                tts = gTTS(text=translated, lang=tgt_tts_lang)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                print("🔊 Translated speech ready, sending to the other device")

                # Send the translated audio to the opposite device
                for device_id, device_websocket in connected_devices.items():
                    if device_websocket != websocket:
                        await device_websocket.send_bytes(buf.read())
                        print(f"🔊 Sent translated audio to device: {device_id}")

            except Exception as e:
                await websocket.send_text(f"TTS failed: {str(e)}")

            # Cleanup
            os.remove(webm_path)
            os.remove(wav_path)

    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected for device: {device_id}.")
        # Remove this device from the connected devices list
        del connected_devices[device_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
