# import gradio as gr
# from transformers import pipeline

# def greet(name):
#     return "Hello " + name
# # We instantiate the Textbox class
# textbox = gr.Textbox(label="Type your name here:", placeholder="John Doe", lines=2)
# gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()
# # demo.launch()


# model = pipeline("text-generation")
#
# def predict(prompt):
#     completion = model(prompt)[0]["generated_text"]
#     return completion
#
# gr.Interface(fn=predict, inputs="text", outputs="text").launch()


# import numpy as np
# import gradio as gr
#
#
# def reverse_audio(audio):
#     sr, data = audio
#     # reversed_audio = (sr, np.flipud(data))
#     reversed_audio = (sr, data)
#     print(f'{type(sr)} - {sr}')
#     print(f'{type(data)} - {data}')
#     return reversed_audio
#
#
# mic = gr.Audio(sources="microphone", type="numpy", label="Speak here...")
# gr.Interface(reverse_audio, mic, "audio").launch()

# import numpy as np
# import gradio as gr
#
# notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
#
#
# def generate_tone(note, octave, duration):
#     sr = 48000
#     a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
#     frequency = a4_freq * 2 ** (tones_from_a4 / 12)
#     duration = int(duration)
#     audio = np.linspace(0, duration, duration * sr)
#     audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
#     return (sr, audio)
#
#
# gr.Interface(
#     generate_tone,
#     [
#         gr.Dropdown(notes, type="index"),
#         gr.Slider(minimum=4, maximum=6, step=1),
#         gr.Number(value=1, label="Duration in seconds"),
#     ],
#     "audio",
# ).launch(share=True)

# ---

# from transformers import pipeline
# import gradio as gr
#
# model = pipeline("automatic-speech-recognition")
#
#
# def transcribe_audio(audio):
#     transcription = model(audio)["text"]
#     return transcription
#
#
# gr.Interface(
#     fn=transcribe_audio,
#     inputs=gr.Audio(type="filepath"),
#     outputs="text",
# ).launch()

# ---

# import gradio as gr
# from transformers import pipeline
#
# model = pipeline("text-generation")
#
# def predict(prompt):
#     completion = model(prompt)[0]["generated_text"]
#     return completion
#
#
# title = "Ask Rick a Question"
# description = """
# The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
# <img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
# """
#
# article = "Check out [the original Rick and Morty Bot](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) that this demo is based off of."
#
# gr.Interface(
#     fn=predict,
#     inputs="textbox",
#     outputs="text",
#     title=title,
#     description=description,
#     article=article,
#     examples=[["What are you doing?"], ["Where should we time travel to?"]],
#     allow_flagging='never',
#     # live=True
# ).launch()

# ---

# import gradio as gr

# title = "GPT-J-6B"
# description = "Gradio Demo for GPT-J 6B, a transformer model trained using Ben Wang's Mesh Transformer JAX. 'GPT-J' refers to the class of model, while '6B' represents the number of trainable parameters. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
# article = "<p style='text-align: center'><a href='https://github.com/kingoflolz/mesh-transformer-jax' target='_blank'>GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model</a></p>"
#
# gr.load(
#     "huggingface/EleutherAI/gpt-j-6B",
#     inputs=gr.Textbox(lines=5, label="Input Text"),
#     title=title,
#     description=description,
#     article=article,
# ).launch()


# gr.load(
#     "spaces/abidlabs/remove-bg", inputs="webcam", title="Remove your webcam background!"
# ).launch()

# ---

# import gradio as gr
#
#
# def flip_text(x):
#     return x[::-1]
#
#
# demo = gr.Blocks()
#
# with demo:
#     gr.Markdown(
#         """
#     # Flip Text!
#     Start typing below to see the output.
#     """
#     )
#     input = gr.Textbox(placeholder="Flip this text")
#     output = gr.Textbox(interactive=True)
#
#     input.submit(fn=flip_text, inputs=input, outputs=output)
#
# demo.launch()


# ---


# import numpy as np
# import gradio as gr
#
# demo = gr.Blocks()
#
#
# def flip_text(x):
#     return x[::-1]
#
#
# def flip_image(x):
#     return np.fliplr(x)
#
#
# with demo:
#     gr.Markdown("Flip text or image files using this demo.")
#     with gr.Tabs():
#         with gr.TabItem("Flip Text"):
#             with gr.Row():
#                 text_input = gr.Textbox()
#                 text_output = gr.Textbox()
#             text_button = gr.Button("Flip")
#         with gr.TabItem("Flip Image"):
#             with gr.Row():
#                 image_input = gr.Image()
#                 image_output = gr.Image()
#             image_button = gr.Button("Flip")
#
#     text_button.click(flip_text, inputs=text_input, outputs=text_output)
#     image_button.click(flip_image, inputs=image_input, outputs=image_output)
#
# demo.launch()


# ---

from transformers import pipeline
import gradio as gr
asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(text):
    return classifier(text)[0]["label"]

demo = gr.Blocks()
with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()
    b1 = gr.Button("Recognize Speech")
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    text.change(text_to_sentiment, inputs=text, outputs=label)
demo.launch()