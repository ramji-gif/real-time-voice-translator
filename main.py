import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
from pydub import AudioSegment
import tempfile
import os
import io

app = FastAPI()
translator = Translator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    src: str
    tgt: str


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

connected_devices = {}

@app.get("/")
def root():
    return {"message": "Backend is working"}

@app.post("/translate-only/")
async def translate_text(req: TranslationRequest):
    try:
        _, _, src_code = language_map.get(req.src, ("hi-IN", "hi", "hi"))
        _, _, tgt_code = language_map.get(req.tgt, ("hi-IN", "hi", "hi"))
        result = translator.translate(req.text, src=src_code, dest=tgt_code).text
        return JSONResponse(content={"translated_text": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/ws/{src}/{tgt}/{device_id}")
async def websocket_endpoint(websocket: WebSocket, src: str, tgt: str, device_id: str):
    await websocket.accept()
    connected_devices[device_id] = websocket

    src_locale, _, src_code = language_map.get(src, ("hi-IN", "hi", "hi"))
    _, tgt_lang, tgt_code = language_map.get(tgt, ("hi-IN", "hi", "hi"))

    recognizer = sr.Recognizer()

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                temp_audio.write(audio_bytes)
                webm_path = temp_audio.name

            wav_path = webm_path.replace(".webm", ".wav")
            AudioSegment.from_file(webm_path).export(wav_path, format="wav")

            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            
            try:
                text = recognizer.recognize_google(audio_data, language=src_locale)
                translated = translator.translate(text, src=src_code, dest=tgt_code).text
                tts = gTTS(text=translated, lang=tgt_lang)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)

                # Send the translated audio to all connected clients
                for d_id, ws in connected_devices.items():
                    if ws != websocket:
                        await ws.send_bytes(buf.read())

            except Exception as e:
                await websocket.send_text(f"Error: {str(e)}")

            os.remove(webm_path)
            os.remove(wav_path)

    except WebSocketDisconnect:
        del connected_devices[device_id]
        print(f"Disconnected: {device_id}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



