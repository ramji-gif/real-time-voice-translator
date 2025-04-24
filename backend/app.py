import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import io
import os

app = FastAPI()
translator = Translator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join("frontend", "index.html"))

language_map = {
    "Hindi":          ("hi-IN", "hi", "hi"),
    "Bengali":        ("bn-IN", "bn", "bn"),
    "Tamil":          ("ta-IN", "ta", "ta"),
    "Telugu":         ("te-IN", "te", "te"),
    "Marathi":        ("mr-IN", "mr", "mr"),
    "Gujarati":       ("gu-IN", "gu", "gu"),
    "Kannada":        ("kn-IN", "kn", "kn"),
    "Malayalam":      ("ml-IN", "ml", "ml"),
    "Urdu":           ("ur-IN", "ur", "ur"),
    "Punjabi":        ("pa-IN", "pa", "pa"),
    "Odia":           ("or-IN", "or", "or"),
    "Assamese":       ("as-IN", "as", "as"),
    "Bhojpuri":       ("hi-IN", "hi", "bho"),
    "Maithili":       ("hi-IN", "hi", "mai"),
    "Chhattisgarhi":  ("hi-IN", "hi", "hne"),
    "Rajasthani":     ("hi-IN", "hi", "raj"),
    "Konkani":        ("hi-IN", "hi", "kok"),
    "Dogri":          ("hi-IN", "hi", "doi"),
    "Kashmiri":       ("hi-IN", "hi", "ks"),
    "Santhali":       ("hi-IN", "hi", "sat"),
    "Sindhi":         ("hi-IN", "hi", "sd"),
    "Manipuri":       ("hi-IN", "hi", "mni"),
    "Bodo":           ("hi-IN", "hi", "brx"),
    "Sanskrit":       ("sa-IN", "sa", "sa")
}

@app.websocket("/ws/{src}/{tgt}")
async def translate_ws(websocket: WebSocket, src: str, tgt: str):
    await websocket.accept()
    recognizer = sr.Recognizer()
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
            stt_locale = language_map.get(src, ("hi-IN", "hi", "hi"))[0]
            try:
                text = recognizer.recognize_google(audio_data, language=stt_locale)
            except sr.UnknownValueError:
                await websocket.send_text("Could not understand audio")
                continue
            src_code = language_map.get(src, ("hi-IN", "hi", "hi"))[2]
            tgt_code = language_map.get(tgt, ("hi-IN", "hi", "hi"))[2]
            translated = translator.translate(text, src=src_code, dest=tgt_code).text
            tts_code = language_map.get(tgt, ("hi-IN", "hi", "hi"))[1]
            tts = gTTS(text=translated, lang=tts_code)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            await websocket.send_bytes(buf.read())
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

   