# SketchSpace — AI-Powered Sketch to Interior Design Generator

> T.Y. BTech CSE (AI & ML) | PE-II Gen AI Lab Mini Project

---

## Overview

SketchSpace transforms rough room layout sketches into photorealistic interior
design images using **ControlNet + Stable Diffusion**. Users upload a hand-drawn
or digital sketch, select an interior aesthetic, and receive 2–4 photorealistic
design variants that preserve the original room structure.

### Key features
- 6 interior design styles (Minimalist, Bohemian, Industrial, Japandi, Luxury, Coastal)
- ControlNet canny edge conditioning for layout preservation
- Automatic CLIP score + structural fidelity evaluation
- Clean web UI with drag-and-drop sketch upload
- Demo mode (no GPU required) + full diffusion mode (CUDA GPU)

---

## Quick Start

### 1. Clone / download the project
```bash
cd sketch_to_design
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

**Demo mode (CPU only — no GPU required):**
```bash
pip install flask pillow numpy torch torchvision werkzeug
```

**Full pipeline (NVIDIA GPU with CUDA):**
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Project Structure

```
sketch_to_design/
├── app.py                  # Flask REST API
├── pipeline.py             # ControlNet + Stable Diffusion pipeline
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   └── evaluation.py       # CLIP score, structural fidelity, FID helpers
├── templates/
│   └── index.html          # Frontend UI
└── static/
    ├── css/main.css
    ├── js/main.js
    ├── uploads/            # Saved input sketches + edge maps
    └── outputs/            # Generated design images
```

---

## How It Works

### Pipeline stages

```
User sketch
    │
    ▼
[1] Canny edge detection (OpenCV)
    → Extracts room structure as binary edge map
    │
    ▼
[2] ControlNet conditioning
    Model: lllyasviel/sd-controlnet-canny
    → Edge map conditions the diffusion process
    │
    ▼
[3] Stable Diffusion v1.5 (Text2Image guided by ControlNet)
    → Style prompt + edge condition → denoising loop
    → UniPC scheduler, 20–50 steps, guidance scale 7–12
    │
    ▼
[4] Output images (512×512, upscalable)
    │
    ▼
[5] Evaluation
    → CLIP score (semantic alignment)
    → Structural fidelity (layout preservation)
```

### Style prompts

Each style uses carefully engineered positive + negative prompts:

| Style | Positive keywords | Negative keywords |
|---|---|---|
| Modern Minimalist | clean lines, neutral, sleek | clutter, dark, maximalist |
| Bohemian | earthy tones, rattan, layered rugs | sterile, cold |
| Industrial Loft | exposed brick, steel, Edison bulbs | colorful, pastel |
| Japandi | natural wood, wabi-sabi, zen | maximalist, harsh |
| Luxury Classical | marble, gold, crystal chandelier | cheap, rustic |
| Coastal | light blue, driftwood, sheer curtains | dark, heavy |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/generate` | Generate design from sketch |
| GET | `/api/styles` | List available styles |
| GET | `/api/history` | Recent generated images |

### POST /api/generate

Form fields:
- `sketch` — image file (required)
- `style` — style id (default: `modern_minimalist`)
- `room_type` — room type string (default: `living room`)
- `custom_prompt` — additional prompt text (optional)
- `num_variants` — 1–4 (default: 3)
- `guidance_scale` — 3–15 (default: 7.5)
- `steps` — 10–50 (default: 20)

---

## Evaluation Metrics

### CLIP Score
Measures semantic similarity between the generated image and the style prompt
using OpenAI's CLIP model (ViT-B/32).

```python
from utils.evaluation import compute_clip_score
score = compute_clip_score("outputs/image.png", "modern minimalist living room...")
# Returns float in [0, 1]
```

### Structural Fidelity
Measures how well the generated image preserves the room layout from the sketch,
by comparing canny edge maps using normalized cross-correlation.

```python
from utils.evaluation import compute_structural_fidelity
score = compute_structural_fidelity("uploads/sketch.png", "outputs/image.png")
# Returns float in [0, 1]
```

### FID Score (for report)
Install pytorch-fid and collect a reference set of real interior images:

```bash
pip install pytorch-fid
python -m pytorch_fid static/reference/ static/outputs/
```

### User Preference Survey
Present 5+ users with: original sketch + 3 generated variants.
Rate each variant 1–5 on:
- Aesthetic quality
- Layout accuracy (does it match the sketch?)
- Style match (does it match the selected style?)

---

## Running on Google Colab (Free GPU)

```python
# In a Colab cell:
!git clone <your-repo-url>
%cd sketch_to_design
!pip install -r requirements.txt
!pip install flask-ngrok pyngrok

from pyngrok import ngrok
import threading

def run_app():
    import subprocess
    subprocess.run(["python", "app.py"])

thread = threading.Thread(target=run_app)
thread.start()

public_url = ngrok.connect(5000)
print("App URL:", public_url)
```

---

## Team Task Split (4 members)

| Member | Responsibility |
|---|---|
| Member 1 | UI (index.html, CSS, JS), sketch upload, style selector |
| Member 2 | Pipeline (pipeline.py), ControlNet integration, edge detection |
| Member 3 | Prompt engineering, style presets, output gallery, API |
| Member 4 | Evaluation (evaluation.py), user survey, report & slides |

---

## Report Outline (6–10 pages)

1. **Introduction** — problem statement, motivation, scope
2. **Literature Review** — GANs vs Diffusion Models, ControlNet paper, prior work
3. **Methodology** — pipeline stages, ControlNet conditioning, prompt engineering
4. **Dataset** — pre-trained model sources, test sketch collection
5. **Implementation** — code architecture, UI screenshots, setup instructions
6. **Results** — CLIP score table, fidelity scores, FID comparison, user survey charts
7. **Discussion** — strengths, limitations (GPU cost, sketch quality sensitivity)
8. **Conclusion & Future Work** — real-time generation, mobile app, 3D rendering

---

## References

1. Zhang, L. et al. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). ICCV.
2. Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models* (Stable Diffusion). CVPR.
3. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML.
4. HuggingFace Diffusers library — https://github.com/huggingface/diffusers
5. lllyasviel/ControlNet — https://github.com/lllyasviel/ControlNet
