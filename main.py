from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import io, tempfile, os
from pydub import AudioSegment

app = FastAPI()
translator = Translator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

clients = []

language_map = {
    language_map = {
    "Hindi": ("hi-IN", "hi-IN", "hi"),
    "English": ("en-IN", "hi-IN", "en"),  # TTS fallback to Hindi
    "Tamil": ("ta-IN", "ta-IN", "ta"),
    "Telugu": ("te-IN", "te-IN", "te"),
    "Bengali": ("bn-IN", "bn-IN", "bn"),
    "Urdu": ("ur-IN", "ur-IN", "ur"),
    "Marathi": ("mr-IN", "mr-IN", "mr"),
    "Gujarati": ("gu-IN", "gu-IN", "gu"),
    "Kannada": ("kn-IN", "kn-IN", "kn"),
    "Malayalam": ("ml-IN", "ml-IN", "ml"),
    "Punjabi": ("pa-IN", "pa-IN", "pa"),
    "Assamese": ("hi-IN", "hi-IN", "as"),
    "Odia": ("hi-IN", "hi-IN", "or"),
    "Bhojpuri": ("hi-IN", "hi-IN", "bho"),
    "Maithili": ("hi-IN", "hi-IN", "mai"),
    "Chhattisgarhi": ("hi-IN", "hi-IN", "hne"),
    "Rajasthani": ("hi-IN", "hi-IN", "raj"),
    "Konkani": ("hi-IN", "hi-IN", "kok"),
    "Dogri": ("hi-IN", "hi-IN", "doi"),
    "Kashmiri": ("hi-IN", "hi-IN", "ks"),
    "Santhali": ("hi-IN", "hi-IN", "sat"),
    "Sindhi": ("hi-IN", "hi-IN", "sd"),
    "Manipuri": ("hi-IN", "hi-IN", "mni"),
    "Bodo": ("hi-IN", "hi-IN", "brx"),
    "Sanskrit": ("hi-IN", "hi-IN", "sa")
}

}

@app.websocket("/ws/{src}/{tgt}")
async def connect(ws: WebSocket, src: str, tgt: str):
    await ws.accept()

    if len(clients) >= 2:
        await ws.send_text("Server already has 2 users connected.")
        await ws.close()
        return

    clients.append({
        "ws": ws,
        "src_locale": language_map.get(src, ("en-US", "en"))[0],
        "src_code": language_map.get(src, ("en-US", "en"))[1],
        "tgt_code": language_map.get(tgt, ("en-US", "en"))[1]
    })

    try:
        while True:
            audio_data = await ws.receive_bytes()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                f.write(audio_data)
                webm_path = f.name

            wav_path = webm_path.replace(".webm", ".wav")
            AudioSegment.from_file(webm_path).export(wav_path, format="wav")

            try:
                r = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio = r.record(source)

                # Get current client's language config
                client = next(c for c in clients if c["ws"] == ws)
                text = r.recognize_google(audio, language=client["src_locale"])
                translated = translator.translate(text, src=client["src_code"], dest=client["tgt_code"]).text
                tts = gTTS(translated, lang=client["tgt_code"])
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)

                # Send to the *other* client
                for c in clients:
                    if c["ws"] != ws:
                        await c["ws"].send_bytes(buf.read())

            except Exception as e:
                await ws.send_text(f"Error: {str(e)}")
            finally:
                os.remove(webm_path)
                os.remove(wav_path)

    except WebSocketDisconnect:
        clients[:] = [c for c in clients if c["ws"] != ws]
