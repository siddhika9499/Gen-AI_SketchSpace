"""
Generation Pipeline — ControlNet + Stable Diffusion
Handles sketch preprocessing, edge detection, and image generation.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch

# ─────────────────────────────────────────────
# Style Definitions
# ─────────────────────────────────────────────
STYLE_PROMPTS = {
    "modern_minimalist": {
        "name": "Modern Minimalist",
        "description": "Clean lines, neutral palette, functional beauty",
        "accent": "#C8B8A2",
        "positive": (
            "modern minimalist interior design, {room_type}, clean lines, "
            "neutral white and beige tones, sleek furniture, open space, "
            "natural light, architectural photography, 8k ultra realistic, "
            "professional interior design magazine"
        ),
        "negative": (
            "clutter, dark, ugly, old, vintage, maximalist, "
            "cartoon, sketch, drawing, watermark, text"
        ),
    },
    "bohemian": {
        "name": "Bohemian / Eclectic",
        "description": "Warm textures, global patterns, layered richness",
        "accent": "#D4856A",
        "positive": (
            "bohemian eclectic interior design, {room_type}, warm earthy tones, "
            "macrame wall art, rattan furniture, layered rugs, lush plants, "
            "colorful throw pillows, cozy atmosphere, soft lighting, "
            "professional interior photography, 8k"
        ),
        "negative": (
            "sterile, cold, minimalist, dark, ugly, cartoon, "
            "sketch, drawing, watermark, text"
        ),
    },
    "industrial_loft": {
        "name": "Industrial Loft",
        "description": "Exposed brick, raw steel, urban sophistication",
        "accent": "#8A8A8A",
        "positive": (
            "industrial loft interior design, {room_type}, exposed brick walls, "
            "concrete floors, steel beams, Edison bulb lighting, dark metal accents, "
            "leather furniture, urban aesthetic, dramatic lighting, "
            "professional interior photography, 8k ultra realistic"
        ),
        "negative": (
            "colorful, pastel, cute, rustic, cartoon, sketch, "
            "drawing, watermark, text, low quality"
        ),
    },
    "japandi": {
        "name": "Japandi / Scandinavian",
        "description": "Wabi-sabi meets Nordic calm — serene, organic",
        "accent": "#B5A898",
        "positive": (
            "japandi scandinavian interior design, {room_type}, natural wood tones, "
            "wabi-sabi aesthetics, minimal zen furniture, shoji screens, "
            "neutral linen textiles, indoor plants, serene atmosphere, "
            "warm soft lighting, professional interior photography, 8k"
        ),
        "negative": (
            "maximalist, colorful, cluttered, dark, harsh, cartoon, "
            "sketch, drawing, watermark, text"
        ),
    },
    "luxury_classic": {
        "name": "Luxury Classical",
        "description": "Marble, gold accents, timeless grandeur",
        "accent": "#C9A84C",
        "positive": (
            "luxury classical interior design, {room_type}, marble floors, "
            "gold accents, crystal chandeliers, rich velvet upholstery, "
            "ornate moldings, symmetrical layout, elegant drapes, "
            "opulent atmosphere, professional interior photography, 8k"
        ),
        "negative": (
            "cheap, rustic, industrial, minimal, cartoon, sketch, "
            "drawing, watermark, text, low quality"
        ),
    },
    "coastal_retreat": {
        "name": "Coastal / Beach",
        "description": "Breezy blues, natural textures, relaxed seaside vibe",
        "accent": "#5F9EA0",
        "positive": (
            "coastal beach interior design, {room_type}, light blue and white palette, "
            "driftwood furniture, woven seagrass rugs, sheer curtains, "
            "nautical accents, bright airy space, ocean-inspired decor, "
            "natural light, professional interior photography, 8k"
        ),
        "negative": (
            "dark, heavy, cluttered, industrial, cartoon, sketch, "
            "drawing, watermark, text"
        ),
    },
}

ROOM_TYPES = [
    "living room", "bedroom", "kitchen", "dining room",
    "home office", "bathroom", "studio apartment", "open plan space"
]


class DesignPipeline:
    """
    Main pipeline for sketch-to-interior-design generation.
    Supports two modes:
      1. Full mode  — ControlNet + Stable Diffusion (requires GPU + diffusers)
      2. Demo mode  — CV-based simulation (works without GPU, for testing/demo)
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.mode = "demo"
        self._try_load_pipeline()

    def _try_load_pipeline(self):
        """Attempt to load the full diffusion pipeline; fall back to demo mode."""
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            from diffusers import UniPCMultistepScheduler

            print("Loading ControlNet model...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            print("Loading Stable Diffusion pipeline...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            )
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_xformers_memory_efficient_attention()
            else:
                self.pipe = self.pipe.to(self.device)

            self.mode = "full"
            print(f"Pipeline loaded in FULL mode on {self.device}")

        except Exception as e:
            print(f"Full pipeline unavailable ({e}). Running in DEMO mode.")
            self.mode = "demo"

    def preprocess_sketch(self, sketch_path: str, session_id: str, output_dir: str):
        """Convert sketch to canny edge map for ControlNet conditioning."""
        img = Image.open(sketch_path).convert("RGB")
        img = img.resize((512, 512), Image.LANCZOS)

        # Convert to grayscale for edge detection
        gray = img.convert("L")

        # Enhance contrast so faint pencil lines are picked up
        gray = ImageEnhance.Contrast(gray).enhance(2.0)

        if self.mode == "full":
            try:
                import cv2
                import numpy as np
                arr = np.array(gray)
                edges = cv2.Canny(arr, threshold1=50, threshold2=150)
                edge_img = Image.fromarray(edges).convert("RGB")
            except ImportError:
                edge_img = self._pil_edge(gray)
        else:
            edge_img = self._pil_edge(gray)

        edge_path = os.path.join(
            os.path.dirname(sketch_path), f"edge_{session_id}.png"
        )
        edge_img.save(edge_path)
        return edge_img, edge_path

    @staticmethod
    def _pil_edge(gray_img):
        """PIL-only edge detection fallback."""
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edges = ImageEnhance.Contrast(edges).enhance(3.0)
        return edges.convert("RGB")

    def build_prompt(self, style_key: str, room_type: str, custom_prompt: str = ""):
        style = STYLE_PROMPTS[style_key]
        positive = style["positive"].format(room_type=room_type)
        if custom_prompt:
            positive = custom_prompt + ", " + positive
        return positive, style["negative"]

    def generate(
        self,
        sketch_path: str,
        style: str,
        room_type: str,
        custom_prompt: str,
        num_variants: int,
        guidance_scale: float,
        steps: int,
        session_id: str,
        output_dir: str,
    ):
        edge_img, _ = self.preprocess_sketch(
            sketch_path, session_id,
            os.path.dirname(sketch_path)
        )
        positive_prompt, negative_prompt = self.build_prompt(style, room_type, custom_prompt)

        results = []
        for i in range(num_variants):
            filename = f"output_{session_id}_v{i+1}.png"
            out_path = os.path.join(output_dir, filename)

            if self.mode == "full":
                seed = 42 + i * 100
                generator = torch.Generator(device=self.device).manual_seed(seed)
                image = self.pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=edge_img,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                image.save(out_path)
            else:
                image = self._demo_generate(edge_img, style, i)
                image.save(out_path)

            results.append({
                'filename': filename,
                'image_path': out_path,
                'style': STYLE_PROMPTS[style]['name'],
                'prompt': positive_prompt,
                'variant': i + 1,
            })

        return results

    def _demo_generate(self, edge_img: Image.Image, style_key: str, variant_idx: int):
        """
        Demo-mode generation: applies style-appropriate color transforms
        to the edge image to simulate a realistic output appearance.
        Used when the full diffusion pipeline is unavailable.
        """
        style_palettes = {
            "modern_minimalist": [(245, 240, 232), (210, 200, 188), (160, 152, 140)],
            "bohemian":          [(210, 140, 100), (180, 120, 80),  (220, 170, 120)],
            "industrial_loft":   [(80,  80,  80),  (120, 110, 100), (60,  58,  55) ],
            "japandi":           [(220, 210, 195), (175, 160, 140), (200, 190, 175)],
            "luxury_classic":    [(240, 225, 190), (200, 170, 100), (230, 215, 180)],
            "coastal_retreat":   [(160, 210, 220), (100, 170, 190), (220, 235, 240)],
        }
        palette = style_palettes.get(style_key, [(200, 190, 180), (160, 150, 140), (220, 210, 200)])
        color = palette[variant_idx % len(palette)]

        base = Image.new("RGB", edge_img.size, color)
        edge_gray = edge_img.convert("L")
        edge_inv = ImageOps.invert(edge_gray).convert("RGB")

        # Blend edge lines into the colored base
        blended = Image.blend(base, edge_inv, alpha=0.35)
        blended = blended.filter(ImageFilter.GaussianBlur(radius=1))
        blended = ImageEnhance.Brightness(blended).enhance(1.05)

        # Add a subtle vignette by pasting a darkened border
        overlay = Image.new("RGB", blended.size, (0, 0, 0))
        mask = Image.new("L", blended.size, 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        w, h = blended.size
        for step in range(60):
            alpha = int(step * 1.5)
            draw.rectangle([step, step, w - step, h - step], outline=alpha)
        blended = Image.composite(overlay, blended, mask)

        return blended
