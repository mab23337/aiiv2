# 🔍 Amazing Image Identifier

A web app that analyzes images using AI — generating captions, detecting and highlighting objects with color-coded bounding boxes, and identifying dominant colors. Built with Flask and deployed on Render.

![Python](https://img.shields.io/badge/python-3.11-blue) ![Flask](https://img.shields.io/badge/flask-3.0-lightgrey) ![Render](https://img.shields.io/badge/deployed-render-purple)

---

## Features

- **AI Captioning** — Describes the image in a natural sentence using [Aya Vision 32B](https://huggingface.co/CohereLabs/aya-vision-32b) via the Hugging Face router
- **Object Detection** — Detects objects using [DETR ResNet-50](https://huggingface.co/facebook/detr-resnet-50), drawing color-coded bounding boxes directly on the image
- **Color Analysis** — Identifies dominant colors in the image using local pixel sampling (no API needed)
- **Read Aloud** — Detected objects can be read aloud using the browser's built-in Text-to-Speech API
- **Session History** — Sidebar tracks previously analyzed images within the current session
- **Download Results** — Export analysis as `.txt` or `.json`
- **Drag & Drop Upload** — Supports JPG and PNG up to 10MB

---

## Tech Stack

| Layer | Technology |
|mab23337|
| Backend | Python 3.11, Flask 3.0 |
| AI — Captioning | Aya Vision 32B (Cohere via HF router) |
| AI — Detection | DETR ResNet-50 (HF Inference API) |
| Color Analysis | Pillow (local, no API) |
| Text-to-Speech | Browser Web Speech API (no API) |
| Deployment | Render (Docker, Starter plan) |

---

## Setup

### Prerequisites
- A free [Hugging Face](https://huggingface.co) account
- A HF API token with read access — [create one here](https://huggingface.co/settings/tokens)

### Local Development

```bash
git clone https://github.com/your-username/amazing-image-identifier.git
cd amazing-image-identifier

pip install -r requirements.txt

export HF_API_TOKEN=hf_your_token_here
export SECRET_KEY=any-random-string

python production_app.py
```

Then open `http://localhost:5000`.

### Deploy to Render

1. Fork or push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com), connecting your repo
3. Render will detect the `Dockerfile` automatically
4. Under **Environment Variables**, add:
   - `HF_API_TOKEN` — your Hugging Face token
   - `SECRET_KEY` — Render can generate this automatically
5. Deploy

The `render.yaml` in the repo configures the service automatically (Starter plan, Virginia region).

---

## Project Structure

```
├── production_app.py      # Flask app — routes, AI calls, image processing
├── Dockerfile             # Container definition
├── render.yaml            # Render deployment config
├── requirements.txt       # Python dependencies
├── templates/
│   ├── index.html         # Main UI with sidebar history
│   └── credits.html       # Open source credits
└── static/
    └── style.css          # Styles
```

---

## Notes

- **Render Starter plan** spins the service down after 15 minutes of inactivity. The first request after that will take ~30 seconds to wake up.
- **Object detection** uses DETR trained on 80 COCO categories (people, animals, vehicles, furniture, etc.). Landscape photos with no recognizable objects will return empty results.
- **Session history** is stored in the browser's `sessionStorage` and clears when the tab is closed. No image data is persisted to disk.
- Uploaded images are deleted from the server immediately after processing.

---

## Open Source Credits

- [Flask](https://flask.palletsprojects.com/)
- [Hugging Face](https://huggingface.co/) — Inference API & models
- [Aya Vision 32B](https://huggingface.co/CohereLabs/aya-vision-32b) by CohereForAI
- [DETR ResNet-50](https://huggingface.co/facebook/detr-resnet-50) by Facebook Research
- [Pillow](https://python-pillow.org/)
- [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk) & [JetBrains Mono](https://www.jetbrains.com/lp/mono/) fonts
