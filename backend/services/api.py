

# ---------------- Install dependencies ----------------
!pip install flask flask-cors cloudinary diffusers transformers accelerate torch torchvision torchaudio pyngrok

!pip install pyngrok


import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import torch
from diffusers import DiffusionPipeline
import cloudinary
import cloudinary.uploader
from pyngrok import ngrok

# ---------------- Ngrok Auth Token ----------------
# Replace YOUR_NGROK_AUTHTOKEN with your actual token

# Remove any leading/trailing spaces
ngrok.set_auth_token("330IaQpARuGEA0BnMP8ODA9D0BJ_3ZYNNqLcod9pmjnBkFQGG")


# ---------------- Flask setup ----------------
app = Flask(__name__)
CORS(app)

cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)

# ---------------- Load Stable Diffusion ----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)
# ---------------- API Endpoint ----------------
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))
        user_id = str(data.get("user_id", "guest"))

        # Generate image
        result = pipe(prompt)
        image = result.images[0]

        # Save to memory
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=prompt[:20].replace(" ", "_"),
            overwrite=True
        )

        return jsonify({"imageUrl": upload_result["secure_url"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask in background ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
print("✅ Flask server started on http://localhost:5003")

# ---------------- Start ngrok tunnel ----------------
public_url = ngrok.connect(5003)
print(f"🌐 ngrok tunnel URL: {public_url}")

# Now you can use `public_url` in your frontend fetch requests:
# fetch(public_url + "/api/generate", { method: "POST", headers:{}, body: JSON.stringify({...}) })

































!pip install flask flask-cors cloudinary diffusers transformers accelerate torch torchvision torchaudio pyngrok


from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import threading
import os
import torch
from diffusers import PixArtSigmaPipeline

import cloudinary
import cloudinary.uploader
from pyngrok import ngrok

# ---------------- Ngrok Auth Token ----------------
# Replace YOUR_NGROK_AUTHTOKEN with your actual token

# Remove any leading/trailing spaces
ngrok.set_auth_token("330IaQpARuGEA0BnMP8ODA9D0BJ_3ZYNNqLcod9pmjnBkFQGG")


# ---------------- Flask setup ----------------
app = Flask(__name__)
CORS(app)

cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)



# Load PixArt model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

# Enable optimizations (safe for Colab)
pipe.enable_attention_slicing()
#pipe.enable_model_cpu_offload()

# ---------------- API Endpoint ----------------
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))
        user_id = str(data.get("user_id", "guest"))

        # Generate image
        result = pipe(prompt)
        image = result.images[0]

        # Save to memory
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=prompt[:20].replace(" ", "_"),
            overwrite=True
        )

        return jsonify({"imageUrl": upload_result["secure_url"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask in background ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
print("✅ Flask server started on http://localhost:5003")

# ---------------- Start ngrok tunnel ----------------
public_url = ngrok.connect(5003)
print(f"🌐 ngrok tunnel URL: {public_url}")

# Now you can use `public_url` in your frontend fetch requests:
# fetch(public_url + "/api/generate", { method: "POST", headers:{}, body: JSON.stringify({...}) })




































































%cd /content
!git clone -b totoro3 https://github.com/camenduru/ComfyUI /content/TotoroUI
%cd /content/TotoroUI

!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --extra-index-url https://download.pytorch.org/whl/cu121



!pip install -q torchsde einops diffusers accelerate xformers==0.0.28.post2
!apt -y install -qq aria2
!pip install flask flask-cors cloudinary  pyngrok

!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors -d /content/TotoroUI/models/unet -o flux1-dev-fp8.safetensors
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /content/TotoroUI/models/vae -o ae.sft
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d /content/TotoroUI/models/clip -o clip_l.safetensors
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors -d /content/TotoroUI/models/clip -o t5xxl_fp8_e4m3fn.safetensors




from flask import Flask, request, jsonify
from flask_cors import CORS
import io, threading, torch, re, random, numpy as np
from PIL import Image
import cloudinary
import cloudinary.uploader
from pyngrok import ngrok

# --- Ngrok setup ---
ngrok.set_auth_token("330EgbQu0bcoKC3uSVPueRFRebb_2zYuUP7QcvKw152ngPNug")

# --- Flask setup ---
app = Flask(__name__)
CORS(app)

cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)

# --- Load FLUX model ---
from totoro_extras import nodes_custom_sampler
from nodes import NODE_CLASS_MAPPINGS
from totoro import model_management

DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model weights
with torch.inference_mode():
    clip = DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
    unet = UNETLoader.load_unet("flux1-dev-fp8.safetensors", "fp8_e4m3fn")[0]
    vae = VAELoader.load_vae("ae.sft")[0]

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    n2 = m * (q + 1) if n * m > 0 else m * (q - 1)
    return n1 if abs(n - n1) < abs(n - n2) else n2

def generate_flux_image(prompt, width=512, height=512, steps=8, sampler_name="euler", scheduler="simple", seed=None):
    if seed is None or seed == 0:
        seed = random.randint(0, 18446744073709551615)

    with torch.no_grad():
        # Encode prompt
        cond, pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
        cond = [[cond, {"pooled_output": pooled}]]

        # Noise
        noise = RandomNoise.get_noise(seed)[0]

        # Guider and sampler
        guider = BasicGuider.get_guider(unet, cond)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]

        # Latent image
        latent_image = EmptyLatentImage.generate(closestNumber(width,16), closestNumber(height,16))[0]
        latent_image = latent_image.detach() if hasattr(latent_image, "detach") else latent_image

        # Sample
        sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        model_management.soft_empty_cache()

        # Decode
        decoded_dict = VAEDecode.decode(vae, sample)[0]  # could be a dict
        if isinstance(decoded_dict, dict) and "sample" in decoded_dict:
            decoded_tensor = decoded_dict["sample"].detach()
        else:
            decoded_tensor = decoded_dict.detach()

        img = Image.fromarray(np.array(decoded_tensor*255, dtype=np.uint8)[0])
    return img

# --- API endpoint ---
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "No prompt provided"}), 400

        prompt = str(data.get("prompt"))
        user_id = str(data.get("user_id", "guest"))

        # Generate FLUX image
        img = generate_flux_image(prompt)

        # Save to memory
        img_io = io.BytesIO()
        img.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # Safe Cloudinary public ID
        safe_id = re.sub(r'\W+', '_', prompt[:30])

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=safe_id,
            overwrite=True
        )

        return jsonify({"imageUrl": upload_result["secure_url"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run Flask ---
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
print("✅ Flask server started on http://localhost:5003")

# --- ngrok tunnel ---
public_url = ngrok.connect(5003)
print(f"🌐 ngrok tunnel URL: {public_url}")




























# ---------------- Install dependencies ----------------
# !pip install flask flask-cors cloudinary diffusers transformers accelerate torch torchvision torchaudio pyngrok

!pip install pyngrok


import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import torch
from diffusers import DiffusionPipeline
import cloudinary
import cloudinary.uploader
from pyngrok import ngrok

# ---------------- Ngrok Auth Token ----------------
# Replace YOUR_NGROK_AUTHTOKEN with your actual token

# Remove any leading/trailing spaces
ngrok.set_auth_token("330I05wJnmiirhaTUh47XKiOQmu_WfdyVPN3S3sjqHJLpVkW")


# ---------------- Flask setup ----------------
app = Flask(__name__)
CORS(app)

cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)

# ---------------- Load Stable Diffusion ----------------
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# ---------------- API Endpoint ----------------
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))
        user_id = str(data.get("user_id", "guest"))

        # Generate image
        result = pipe(prompt)
        image = result.images[0]

        # Save to memory
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=prompt[:20].replace(" ", "_"),
            overwrite=True
        )

        return jsonify({"imageUrl": upload_result["secure_url"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask in background ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
print("✅ Flask server started on http://localhost:5003")

# ---------------- Start ngrok tunnel ----------------
public_url = ngrok.connect(5003)
print(f"🌐 ngrok tunnel URL: {public_url}")

# Now you can use `public_url` in your frontend fetch requests:
# fetch(public_url + "/api/generate", { method: "POST", headers:{}, body: JSON.stringify({...}) })





















# Install dependencies
!pip install flask-cors cloudinary
!pip install diffusers transformers accelerate
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
!npm install -g localtunnel

from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import threading
import os
import torch
from diffusers import DiffusionPipeline
import cloudinary
import cloudinary.uploader

# ✅ fixed here


app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "ngrok-skip-browser-warning"]
    }
})

# ---------------- Cloudinary Configuration ----------------
cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)

# ---------------- Load model ----------------
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# ---------------- Generate and Upload Endpoint ----------------
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))
        user_id = str(data.get("user_id", "guest"))  # optional, for user folders

        # 1️⃣ Generate image
        result = pipe(prompt)
        image = result.images[0]

        # 2️⃣ Save to memory as JPEG
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # 3️⃣ Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=prompt[:20].replace(" ", "_"),
            overwrite=True
        )

        # 4️⃣ Get public URL
        image_url = upload_result["secure_url"]

        # 5️⃣ Return URL to frontend
        return jsonify({"imageUrl": image_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

thread = threading.Thread(target=run_flask)
thread.start()

# Start LocalTunnel (✅ make sure port matches 5003 everywhere)
os.system("lt --port 5003 --print-requests")
!lt --port 5003 --subdomain myflaskdemo























































# ---------------- Install dependencies ----------------
!pip install flask flask-cors cloudinary diffusers transformers accelerate torch torchvision torchaudio pyngrok

!pip install pyngrok


import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import torch
from diffusers import DiffusionPipeline
import cloudinary
import cloudinary.uploader
from pyngrok import ngrok

# ---------------- Ngrok Auth Token ----------------
# Replace YOUR_NGROK_AUTHTOKEN with your actual token

# Remove any leading/trailing spaces
ngrok.set_auth_token("31XtNzi969M5yp5EkNl1Qk3jQna_5TQ9uo2XNmRncB3NRTW4n")


# ---------------- Flask setup ----------------
app = Flask(__name__)
CORS(app)

cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)

# ---------------- Load Stable Diffusion ----------------
model_id = "stablediffusionapi/deliberate-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# ---------------- API Endpoint ----------------
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))
        user_id = str(data.get("user_id", "guest"))

        # Generate image
        result = pipe(prompt)
        image = result.images[0]

        # Save to memory
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=prompt[:20].replace(" ", "_"),
            overwrite=True
        )

        return jsonify({"imageUrl": upload_result["secure_url"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask in background ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
print("✅ Flask server started on http://localhost:5003")

# ---------------- Start ngrok tunnel ----------------
public_url = ngrok.connect(5003)
print(f"🌐 ngrok tunnel URL: {public_url}")

# Now you can use `public_url` in your frontend fetch requests:
# fetch(public_url + "/api/generate", { method: "POST", headers:{}, body: JSON.stringify({...}) })























# ---------------- Install dependencies ----------------
!pip install flask flask-cors cloudinary diffusers transformers accelerate torch torchvision torchaudio pyngrok

!pip install pyngrok


import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import torch
from diffusers import DiffusionPipeline
import cloudinary
import cloudinary.uploader
from pyngrok import ngrok

# ---------------- Ngrok Auth Token ----------------
# Replace YOUR_NGROK_AUTHTOKEN with your actual token

# Remove any leading/trailing spaces
ngrok.set_auth_token("31XtNzi969M5yp5EkNl1Qk3jQna_5TQ9uo2XNmRncB3NRTW4n")


# ---------------- Flask setup ----------------
app = Flask(__name__)
CORS(app)

cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)

# ---------------- Load Stable Diffusion ----------------
model_id = "gsdf/Counterfeit-V2.5"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# ---------------- API Endpoint ----------------
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))
        user_id = str(data.get("user_id", "guest"))

        # Generate image
        result = pipe(prompt)
        image = result.images[0]

        # Save to memory
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=prompt[:20].replace(" ", "_"),
            overwrite=True
        )

        return jsonify({"imageUrl": upload_result["secure_url"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask in background ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
print("✅ Flask server started on http://localhost:5003")

# ---------------- Start ngrok tunnel ----------------
public_url = ngrok.connect(5003)
print(f"🌐 ngrok tunnel URL: {public_url}")

# Now you can use `public_url` in your frontend fetch requests:
# fetch(public_url + "/api/generate", { method: "POST", headers:{}, body: JSON.stringify({...}) })























# Install dependencies
!pip install flask-cors
!npm install -g localtunnel
%pip install diffusers transformers accelerate safetensors sentencepiece
%pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import threading
import os
import torch
from diffusers import PixArtSigmaPipeline

app = Flask(__name__)
CORS(app)

# Load PixArt model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

# Enable optimizations (safe for Colab)
pipe.enable_attention_slicing()
#pipe.enable_model_cpu_offload()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))

        # Generate image
        result = pipe(prompt)
        image = result.images[0]

        # Save to memory as JPEG (smaller & faster for web)
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg", as_attachment=True, download_name="image.jpg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server in a separate thread
def run_flask():
    app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False)

thread = threading.Thread(target=run_flask)
thread.start()

# Start LocalTunnel and print public URL
os.system("lt --port 5002 --print-requests")
!lt --port 5002 --subdomain myflaskdemo





























# Install dependencies
!pip install flask-cors
!npm install -g localtunnel
%pip install diffusers transformers accelerate
%pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import threading
import os
import torch
from diffusers import DiffusionPipeline

app = Flask(__name__)
CORS(app)

# Load model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))

        # Generate image
        result = pipe(prompt)
        image = result.images[0]

        # Save to memory as JPEG
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)  # JPEG format, quality=90
        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg", as_attachment=True, download_name="image.jpg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server in a separate thread
def run_flask():
    app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False)

thread = threading.Thread(target=run_flask)
thread.start()

# Start LocalTunnel and print public URL
os.system("lt --port 5002 --print-requests")
!lt --port 5002 --subdomain myflaskdemo
































# Install dependencies
!pip install flask-cors cloudinary
!pip install diffusers transformers accelerate
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
!npm install -g localtunnel

from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import threading
import os
import torch
from diffusers import DiffusionPipeline
import cloudinary
import cloudinary.uploader

# ✅ fixed here
app = Flask(__name__)
CORS(app)

# ---------------- Cloudinary Configuration ----------------
cloudinary.config(
    cloud_name="dr6yhnu0z",
    api_key="684599771452817",
    api_secret="ACpDif1_lkgVJwFkIgVaItR8qGA",
    secure=True
)

# ---------------- Load model ----------------
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# ---------------- Generate and Upload Endpoint ----------------
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = str(data.get("prompt", "A default prompt"))
        user_id = str(data.get("user_id", "guest"))  # optional, for user folders

        # 1️⃣ Generate image
        result = pipe(prompt)
        image = result.images[0]

        # 2️⃣ Save to memory as JPEG
        img_io = io.BytesIO()
        image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        # 3️⃣ Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            folder=f"users/{user_id}",
            public_id=prompt[:20].replace(" ", "_"),
            overwrite=True
        )

        # 4️⃣ Get public URL
        image_url = upload_result["secure_url"]

        # 5️⃣ Return URL to frontend
        return jsonify({"imageUrl": image_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run Flask ----------------
def run_flask():
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)

thread = threading.Thread(target=run_flask)
thread.start()

# Start LocalTunnel (✅ make sure port matches 5003 everywhere)
os.system("lt --port 5003 --print-requests")
!lt --port 5003 --subdomain myflaskdemo
