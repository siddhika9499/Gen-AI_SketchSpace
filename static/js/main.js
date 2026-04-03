/* SketchSpace — main.js */

let selectedStyle = 'modern_minimalist';
let selectedFile = null;

// ── Init ─────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadStyles();
  checkMode();
  initDropZone();
  initFileInput();
});

// ── Mode check ───────────────────────────
async function checkMode() {
  const badge = document.getElementById('mode-badge');
  try {
    const r = await fetch('/api/styles');
    if (r.ok) {
      badge.textContent = 'API online';
      badge.className = 'mode-badge full';
    }
  } catch {
    badge.textContent = 'Demo mode';
    badge.className = 'mode-badge demo';
  }
}

// ── Load Styles ──────────────────────────
async function loadStyles() {
  try {
    const r = await fetch('/api/styles');
    const styles = await r.json();
    renderStyleGrid(styles);
  } catch {
    const fallback = [
      { id: 'modern_minimalist', name: 'Modern Minimalist', description: 'Clean lines, neutral palette', accent: '#C8B8A2' },
      { id: 'bohemian',          name: 'Bohemian / Eclectic', description: 'Warm textures, global patterns', accent: '#D4856A' },
      { id: 'industrial_loft',   name: 'Industrial Loft',    description: 'Exposed brick, raw steel',  accent: '#8A8A8A' },
      { id: 'japandi',           name: 'Japandi / Scandinavian', description: 'Wabi-sabi meets Nordic calm', accent: '#B5A898' },
      { id: 'luxury_classic',    name: 'Luxury Classical',   description: 'Marble, gold, timeless grandeur', accent: '#C9A84C' },
      { id: 'coastal_retreat',   name: 'Coastal / Beach',    description: 'Breezy blues, relaxed vibe', accent: '#5F9EA0' },
    ];
    renderStyleGrid(fallback);
  }
}

function renderStyleGrid(styles) {
  const grid = document.getElementById('styleGrid');
  grid.innerHTML = '';
  styles.forEach(s => {
    const card = document.createElement('div');
    card.className = 'style-card' + (s.id === selectedStyle ? ' active' : '');
    card.style.setProperty('--swatch', s.accent);
    card.innerHTML = `
      <div class="style-name">${s.name}</div>
      <div class="style-desc">${s.description}</div>
    `;
    card.onclick = () => {
      selectedStyle = s.id;
      document.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
    };
    grid.appendChild(card);
  });
}

// ── Drop Zone ────────────────────────────
function initDropZone() {
  const zone = document.getElementById('dropZone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) setFile(file);
  });
  zone.addEventListener('click', e => {
    if (e.target.closest('.btn-browse')) return;
    document.getElementById('fileInput').click();
  });
}

function initFileInput() {
  document.getElementById('fileInput').addEventListener('change', e => {
    if (e.target.files[0]) setFile(e.target.files[0]);
  });
}

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('previewImg');
    img.src = e.target.result;
    img.classList.add('visible');
    document.getElementById('dropInner').style.display = 'none';
  };
  reader.readAsDataURL(file);
}

// ── Generate ─────────────────────────────
async function generateDesign() {
  if (!selectedFile) {
    alert('Please upload a sketch first.');
    return;
  }

  const btn = document.getElementById('generateBtn');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Generating...';

  showLoadingState();

  const form = new FormData();
  form.append('sketch', selectedFile);
  form.append('style', selectedStyle);
  form.append('room_type', document.getElementById('roomType').value);
  form.append('custom_prompt', document.getElementById('customPrompt').value);
  form.append('num_variants', document.getElementById('numVariants').value);
  form.append('guidance_scale', document.getElementById('guidanceScale').value);
  form.append('steps', document.getElementById('inferenceSteps').value);

  const loadingMessages = [
    ['Preprocessing sketch...', 'ControlNet is detecting room structure'],
    ['Running edge detection...', 'Canny algorithm extracting contours'],
    ['Conditioning diffusion model...', 'Stable Diffusion is initialising'],
    ['Generating design variants...', 'Denoising across inference steps'],
    ['Computing evaluation metrics...', 'Scoring CLIP alignment and fidelity'],
  ];
  let msgIdx = 0;
  const msgInterval = setInterval(() => {
    msgIdx = (msgIdx + 1) % loadingMessages.length;
    document.getElementById('loadingText').textContent = loadingMessages[msgIdx][0];
    document.querySelector('.loading-sub').textContent = loadingMessages[msgIdx][1];
  }, 2200);

  try {
    const res = await fetch('/api/generate', { method: 'POST', body: form });
    clearInterval(msgInterval);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Generation failed');
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    clearInterval(msgInterval);
    alert('Error: ' + err.message);
    hideLoadingState();
  } finally {
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Generate designs';
  }
}

// ── State Management ──────────────────────
function showLoadingState() {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';
  document.getElementById('loadingState').style.display = 'block';
  document.getElementById('sketchRow').style.display = 'none';
  document.getElementById('gallery').innerHTML = '';
  document.getElementById('metricsGrid').innerHTML = '';
  document.getElementById('resultsContent').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideLoadingState() {
  document.getElementById('loadingState').style.display = 'none';
  document.getElementById('resultsPlaceholder').style.display = 'flex';
  document.getElementById('resultsContent').style.display = 'none';
}

// ── Render Results ────────────────────────
function renderResults(data) {
  document.getElementById('loadingState').style.display = 'none';

  // Sketch + edge row
  const sketchRow = document.getElementById('sketchRow');
  document.getElementById('sketchThumb').src = data.sketch_url + '?t=' + Date.now();
  document.getElementById('edgeThumb').src = data.edge_url + '?t=' + Date.now();
  sketchRow.style.display = 'flex';

  // Gallery
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = '';
  data.results.forEach((r, i) => {
    const item = document.createElement('div');
    item.className = 'gallery-item';
    item.style.animationDelay = (i * 0.1) + 's';
    item.innerHTML = `
      <img class="gallery-img" src="${r.url}?t=${Date.now()}" alt="Design variant ${r.variant}"/>
      <div class="gallery-info">
        <div class="gallery-style">${r.style}</div>
        <div class="gallery-variant">Variant ${r.variant}</div>
        <div class="gallery-scores">
          <span class="score-pill">CLIP: ${r.clip_score.toFixed(3)}</span>
          <span class="score-pill">Fidelity: ${r.structural_fidelity.toFixed(3)}</span>
        </div>
        <div class="gallery-prompt">${r.prompt}</div>
        <div class="gallery-actions">
          <a class="btn-download" href="${r.url}" download="design_v${r.variant}.png">Download</a>
        </div>
      </div>
    `;
    gallery.appendChild(item);
  });

  // Metrics
  renderMetrics(data);
}

function renderMetrics(data) {
  const grid = document.getElementById('metricsGrid');
  grid.innerHTML = '';

  const avgClip = data.results.reduce((s, r) => s + r.clip_score, 0) / data.results.length;
  const avgFidelity = data.results.reduce((s, r) => s + r.structural_fidelity, 0) / data.results.length;

  const metrics = [
    { name: 'Variants generated', value: data.results.length, sub: 'output images' },
    { name: 'Avg CLIP score', value: avgClip.toFixed(3), sub: 'prompt-image alignment' },
    { name: 'Avg fidelity', value: avgFidelity.toFixed(3), sub: 'layout preservation' },
    { name: 'Style', value: data.results[0]?.style || '—', sub: data.room_type },
  ];

  metrics.forEach(m => {
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-name">${m.name}</div>
      <div class="metric-value">${m.value}</div>
      <div class="metric-sub">${m.sub}</div>
    `;
    grid.appendChild(card);
  });
}
