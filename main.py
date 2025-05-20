import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator, LANGUAGES
import io
import os
import json
from pydub import AudioSegment
import tempfile

app = FastAPI()
translator = Translator()
connected_devices = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
def root():
    return {"message": "✅ Voice Translator backend is running."}

@app.post("/translate-only/")
async def translate_only(req: BaseModel):
    body = await req.json()
    text = body["text"]
    source_lang = body["source_lang"]
    target_lang = body["target_lang"]
    
    _, _, src_code = language_map.get(source_lang, ("hi-IN", "hi", "hi"))
    _, _, tgt_code = language_map.get(target_lang, ("hi-IN", "hi", "hi"))

    try:
        result = translator.translate(text, src=src_code, dest=tgt_code).text
        return {"translated_text": result}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/{src}/{tgt}/{device_id}")
async def websocket_endpoint(websocket: WebSocket, src: str, tgt: str, device_id: str):
    await websocket.accept()
    print(f"🔗 Connected: {device_id} ({src} → {tgt})")

    src_locale, _, src_code = language_map.get(src, ("hi-IN", "hi", "hi"))
    _, tgt_tts_lang, tgt_code = language_map.get(tgt, ("hi-IN", "hi", "hi"))

    recognizer = sr.Recognizer()
    connected_devices[device_id] = websocket

    try:
        while True:
            data = await websocket.receive_bytes()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as webm_file:
                webm_file.write(data)
                webm_path = webm_file.name

            wav_path = webm_path.replace(".webm", ".wav")

            try:
                AudioSegment.from_file(webm_path).export(wav_path, format="wav")
                with sr.AudioFile(wav_path) as source:
                    audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language=src_locale)
                print("🗣 Recognized:", text)
            except Exception as e:
                await websocket.send_text(f"STT failed: {str(e)}")
                continue
            finally:
                os.remove(webm_path)
                os.remove(wav_path)

            try:
                translated = translator.translate(text, src=src_code, dest=tgt_code).text
                print("🌍 Translated:", translated)
            except Exception as e:
                await websocket.send_text(f"Translation failed: {str(e)}")
                continue

            try:
                tts = gTTS(text=translated, lang=tgt_tts_lang)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                await websocket.send_bytes(buf.read())
            except Exception as e:
                await websocket.send_text(f"TTS failed: {str(e)}")

    except WebSocketDisconnect:
        print(f"❌ Disconnected: {device_id}")
        connected_devices.pop(device_id, None)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

