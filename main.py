from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO

import getpass
import os
import threading

from pyngrok import ngrok, conf

# Load model
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
conf.get_default().auth_token = getpass.getpass()

# Start flask app and set to ngrok
app = Flask(__name__)
# run_with_ngrok(app)
port = "5000"

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

# ... Update inbound traffic via APIs to use the public-facing ngrok URL

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  image = pipe(prompt).images[0]
  print("Image generated! Converting image ...")
  
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  print("Sending image ...")
  return render_template('index.html', generated_image=img_str)


# if __name__ == '__main__':
#     app.run()

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()