import os
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

import os
import gradio as gr
import whisper
from gtts import gTTS
import io
from transformers import pipeline
from groq import Groq

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load the Whisper model
whisper_model = whisper.load_model("base")  # You can choose other models like "small", "medium", "large"

# Initialize the grammar correction pipeline
corrector = pipeline("text2text-generation", model="pszemraj/flan-t5-large-grammar-synthesis")

def process_audio(file_path):
    try:
        # Load the audio file
        audio = whisper.load_audio(file_path)

        # Transcribe the audio using Whisper
        result = whisper_model.transcribe(audio)
        user_text = result["text"]

        # Display the user input text
        corrected_text = corrector(user_text)[0]['generated_text'].strip()

        # Generate a response using Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": corrected_text}],
            model="llama3-8b-8192",  # Replace with the correct model if necessary
        )

        # Access the response using dot notation
        response_message = chat_completion.choices[0].message.content.strip()

        # Convert the response text to speech
        tts = gTTS(response_message)
        response_audio_io = io.BytesIO()
        tts.write_to_fp(response_audio_io)  # Save the audio to the BytesIO object
        response_audio_io.seek(0)

        # Save audio to a file to ensure it's generated correctly
        with open("response.mp3", "wb") as audio_file:
            audio_file.write(response_audio_io.getvalue())

        # Return the original text, corrected text, and the path to the saved audio file
        return user_text, corrected_text, "response.mp3"

    except Exception as e:
        return f"An error occurred: {e}", None, None

# Create a Gradio interface with a submit button
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),  # Use type="filepath"
    outputs=[
        gr.Textbox(label="User voice input into text"),  # Original user input text
        gr.Textbox(label="Corrected version of user input"),  # Corrected text
        gr.Audio(label="Response Audio")  # Response audio
    ],
    live=False,  # Ensure live mode is off to use a submit button
    title="Audio Processing with Grammar Correction",
    description="Upload an audio file, which will be transcribed, corrected for grammar, and then used to generate a response.",
    allow_flagging="never"
)

iface.launch()
