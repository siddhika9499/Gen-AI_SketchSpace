"""
AI-Powered Sketch to Interior Design Generator
Main Flask Application
"""

import os
import uuid
import json
import time
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pipeline import DesignPipeline
from utils.evaluation import compute_clip_score, compute_structural_fidelity

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

pipeline = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = DesignPipeline()
    return pipeline


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """Main generation endpoint."""
    try:
        sketch_file = request.files.get('sketch')
        style = request.form.get('style', 'modern_minimalist')
        room_type = request.form.get('room_type', 'living room')
        custom_prompt = request.form.get('custom_prompt', '')
        num_variants = int(request.form.get('num_variants', 3))
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        steps = int(request.form.get('steps', 20))

        if not sketch_file or not allowed_file(sketch_file.filename):
            return jsonify({'error': 'Invalid or missing sketch file'}), 400

        session_id = str(uuid.uuid4())[:8]
        sketch_filename = f"sketch_{session_id}.png"
        sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], sketch_filename)
        sketch_file.save(sketch_path)

        pipe = get_pipeline()
        results = pipe.generate(
            sketch_path=sketch_path,
            style=style,
            room_type=room_type,
            custom_prompt=custom_prompt,
            num_variants=num_variants,
            guidance_scale=guidance_scale,
            steps=steps,
            session_id=session_id,
            output_dir=app.config['OUTPUT_FOLDER']
        )

        # Compute evaluation metrics
        metrics = []
        for r in results:
            clip_score = compute_clip_score(r['image_path'], r['prompt'])
            fidelity = compute_structural_fidelity(sketch_path, r['image_path'])
            metrics.append({
                'clip_score': round(clip_score, 3),
                'structural_fidelity': round(fidelity, 3),
            })

        response_data = {
            'session_id': session_id,
            'sketch_url': f"/static/uploads/{sketch_filename}",
            'edge_url': f"/static/uploads/edge_{session_id}.png",
            'results': [
                {
                    'url': f"/static/outputs/{r['filename']}",
                    'style': r['style'],
                    'prompt': r['prompt'],
                    'variant': i + 1,
                    'clip_score': metrics[i]['clip_score'],
                    'structural_fidelity': metrics[i]['structural_fidelity'],
                }
                for i, r in enumerate(results)
            ],
            'style': style,
            'room_type': room_type,
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/styles', methods=['GET'])
def get_styles():
    """Return available interior styles."""
    from pipeline import STYLE_PROMPTS
    styles = []
    for key, data in STYLE_PROMPTS.items():
        styles.append({
            'id': key,
            'name': data['name'],
            'description': data['description'],
            'accent': data['accent'],
        })
    return jsonify(styles)


@app.route('/api/history', methods=['GET'])
def get_history():
    """Return generated image history."""
    outputs = Path(app.config['OUTPUT_FOLDER'])
    files = sorted(outputs.glob('*.png'), key=os.path.getmtime, reverse=True)[:20]
    history = []
    for f in files:
        if not f.stem.startswith('edge_'):
            history.append({
                'url': f"/static/outputs/{f.name}",
                'name': f.stem,
                'created': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))
            })
    return jsonify(history)


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/outputs', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
