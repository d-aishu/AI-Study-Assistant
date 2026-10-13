"""
Voice output: converts assistant answers to spoken audio (MP3 bytes) using
gTTS (Google Text-to-Speech). No API key required.

Voice input (microphone -> text) is handled directly in app.py via
streamlit_mic_recorder.speech_to_text, which already wraps recording +
Google's speech-recognition API in a single widget.
"""
import io

from gtts import gTTS


def text_to_speech(text: str, lang: str = "en") -> bytes:
    """
    Convert text to spoken audio and return raw MP3 bytes, ready to hand
    to st.audio(). Truncates very long answers so TTS stays fast — the
    full text answer remains visible on screen either way.
    """
    if not text or not text.strip():
        raise ValueError("Nothing to speak — the answer was empty.")

    speakable = text.strip()
    if len(speakable) > 3000:
        speakable = speakable[:3000] + ". Truncated. See the full answer on screen."

    buf = io.BytesIO()
    gTTS(text=speakable, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf.read()
