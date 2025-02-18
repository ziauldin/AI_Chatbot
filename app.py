!pip install gradio
!pip install groq
!pip install soundfile

import gradio as gr
import groq
import io
import numpy as np
import soundfile as sf
import os

# Hardcoded API key (For production, set it as an environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

def transcribe_audio(audio):
    if audio is None:
        return "No audio detected."

    client = groq.Client(api_key=GROQ_API_KEY)

    # Convert audio to the required format
    audio_data = audio[1]  # Get numpy array
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, audio[0], format='wav')
    buffer.seek(0)

    try:
        # Use Distil-Whisper for transcription
        completion = client.audio.transcriptions.create(
            model="distil-whisper-large-v3-en",
            file=("audio.wav", buffer),
            response_format="text"
        )
        return completion
    except Exception as e:
        return f"Error in transcription: {str(e)}"

def generate_response(transcription):
    if not transcription:
        return "No transcription available. Please try speaking again."

    client = groq.Client(api_key=GROQ_API_KEY)

    try:
        # Use Llama 3 70B for response generation
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in response generation: {str(e)}"

def process_audio(audio):
    transcription = transcribe_audio(audio)
    response = generate_response(transcription)
    return transcription, response

# Custom CSS for UI
custom_css = """
/* General container styling */
.gradio-container {
    background: linear-gradient(to right, #1e1e2f, #2a2a40);
    color: white;
    font-family: 'Poppins', sans-serif;
    padding: 20px;
    border-radius: 10px;
}

/* Title Styling */
h1, h2, h3 {
    font-weight: 700;
    text-align: center;
}

/* Primary button */
.gr-button-primary {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border: none;
    color: white !important;
    font-size: 16px;
    padding: 12px 20px;
    border-radius: 8px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 4px 10px rgba(255, 75, 43, 0.3);
}

.gr-button-primary:hover {
    background: linear-gradient(135deg, #ff4b2b, #ff416c);
    transform: scale(1.05);
}

/* Secondary button */
.gr-button-secondary {
    background: transparent;
    border: 2px solid #ff4b2b;
    color: #ff4b2b !important;
    font-size: 16px;
    padding: 10px 18px;
    border-radius: 8px;
    transition: all 0.3s ease-in-out;
}

.gr-button-secondary:hover {
    background: #ff4b2b;
    color: white !important;
    transform: scale(1.05);
}

/* Audio input styling */
.gr-audio {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    padding: 10px;
    border-radius: 8px;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease-in-out;
}

.gr-audio:hover {
    border-color: #ff4b2b;
    transform: scale(1.02);
}

/* Textbox styling */
.gr-textbox {
    background: rgba(255, 255, 255, 0.1);
    border: none;
    padding: 12px;
    border-radius: 8px;
    color: white;
    transition: all 0.3s ease-in-out;
    font-size: 14px;
}

.gr-textbox:focus {
    border: 2px solid #ff4b2b;
    box-shadow: 0px 0px 10px rgba(255, 75, 43, 0.4);
}

/* Badge styling */
#groq-badge {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
    background: rgba(255, 75, 43, 0.9);
    padding: 10px 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(255, 75, 43, 0.3);
    font-size: 14px;
    color: white;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}

#groq-badge:hover {
    background: rgba(255, 75, 43, 1);
    transform: scale(1.05);
}

/* Input field placeholders */
.gr-textbox::placeholder {
    color: rgba(255, 255, 255, 0.5);
}
"""

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎙️ AI Voice Assistant")

    with gr.Row():
        audio_input = gr.Audio(label="Speak!", type="numpy")

    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription")
        response_output = gr.Textbox(label="AI Assistant Response")

    submit_button = gr.Button("Process", variant="primary")

    gr.HTML("""
    <div id="groq-badge">
        <div style="color: #f55036; font-weight: bold;">BUILD BY ZIA</div>
    </div>
    """)

    submit_button.click(
        process_audio,
        inputs=[audio_input],
        outputs=[transcription_output, response_output]
    )

    gr.Markdown("""
    ## How to use this app:
    1. Click on the microphone icon and speak your message or upload an audio file.
    2. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, and webm.
    3. Click the "Process" button to transcribe your speech and generate a response.
    4. The transcription and AI assistant response will be displayed.
    """)

demo.launch()
