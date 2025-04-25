import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Voice Translator backend is running."}


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
    
    buffer = bytearray()
    chunk_count = 0
    CHUNKS_BEFORE_PROCESSING = 6  # ~2 seconds if 300ms chunks
    src_locale, tts_lang, src_code = language_map.get(src, ("hi-IN", "hi", "hi"))
    _, _, tgt_code = language_map.get(tgt, ("hi-IN", "hi", "hi"))

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            buffer.extend(audio_chunk)
            chunk_count += 1

            if chunk_count >= CHUNKS_BEFORE_PROCESSING:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as webm_file:
                    webm_file.write(buffer)
                    webm_path = webm_file.name

                wav_path = webm_path.replace(".webm", ".wav")
                try:
                    AudioSegment.from_file(webm_path).export(wav_path, format="wav")
                except Exception as e:
                    await websocket.send_text(f"Audio conversion failed: {str(e)}")
                    os.remove(webm_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                try:
                    with sr.AudioFile(wav_path) as source:
                        audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language=src_locale)
                except sr.UnknownValueError:
                    await websocket.send_text("Could not understand audio")
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue
                except sr.RequestError as e:
                    await websocket.send_text(f"Speech recognition failed: {e}")
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                try:
                    translated = translator.translate(text, src=src_code, dest=tgt_code).text
                except Exception as e:
                    await websocket.send_text(f"Translation failed: {str(e)}")
                    os.remove(webm_path)
                    os.remove(wav_path)
                    buffer.clear()
                    chunk_count = 0
                    continue

                try:
                    tts = gTTS(text=translated, lang=tts_lang)
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    await websocket.send_bytes(buf.read())
                except Exception as e:
                    await websocket.send_text(f"TTS failed: {str(e)}")

                # Clean up and reset
                os.remove(webm_path)
                os.remove(wav_path)
                buffer.clear()
                chunk_count = 0

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
 




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
 


   
        
        
