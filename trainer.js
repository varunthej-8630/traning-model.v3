/* ═══════════════════════════════════════════════════════════
   CUSTOS Trainer v3.2 — Advanced Guardian Intelligence
   Engineered for Multi-Class, High-Performance ML Training
   ═══════════════════════════════════════════════════════════ */
'use strict';

/* ── CLASS PALETTE ──────────────────────────────────────── */
const PALETTE = [
  '#00e5ff','#00ff88','#ffd600','#ff3860',
  '#a855f7','#fb923c','#34d399','#f472b6',
  '#60a5fa','#facc15'
];

/* ── STATE ──────────────────────────────────────────────── */
let classes = [
  { id: 0, label: 'class_1', samples: [], previews: [], color: PALETTE[0] },
  { id: 1, label: 'class_2', samples: [], previews: [], color: PALETTE[1] }
];
let nextClassId   = 2;
let currentMode   = 'webcam';
let isRecording   = false;
let recordingCId  = null;
let recordInterval= null;
let model         = null;
let mobileNet     = null;
let mobileNetReady= false;
let distChart     = null;
let metricsChart  = null;
let predInterval  = null;

/* ── DOM HELPERS ────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const video          = $('video');
const canvas         = $('canvas');
const ctx            = canvas.getContext('2d');
const classManager   = $('class-manager');
const webcamRecBtns  = $('webcam-rec-btns');
const uploadZone     = $('upload-zone');
const uploadClassSel = $('upload-class-select');
const thumbGrid      = $('thumb-grid');
const procLog        = $('proc-log');
const statTotal      = $('stat-total');
const statClasses    = $('stat-classes');
const statBalance    = $('stat-balance');
const confFill       = $('conf-fill');
const confReadout    = $('conf-readout');
const confPrediction = $('conf-prediction');
const progressWrap   = $('progress-wrap');
const progressFill   = $('progress-fill');
const progressLabel  = $('progress-label');
const btnTrain       = $('btn-train');
const trainStatus    = $('train-status');
const modelLoadStatus= $('model-load-status');
const cfgEpochs      = $('cfg-epochs');
const cfgEpochsVal   = $('cfg-epochs-val');
const cfgValSplit    = $('cfg-val-split');
const cfgValSplitVal = $('cfg-val-split-val');
const cfgLR          = $('cfg-lr');

/* ══════════════════════════════════════════════════════════
   MOBILENET ENGINE
 ══════════════════════════════════════════════════════════ */
async function loadMobileNet() {
  const btn = $('ob-start');
  try {
    trainStatus.textContent = 'SYSTEM: LOADING MOBILENET v2...';
    mobileNet = await mobilenet.load({ version: 2, alpha: 0.5 });
    mobileNetReady = true;
    trainStatus.textContent = 'SYSTEM: READY. COLLECT DATA.';
    console.log('MobileNet V2 Load Success');
    if (btn) {
      btn.innerHTML = 'LAUNCH SYSTEM <i class="fa-solid fa-arrow-right"></i>';
      btn.style.opacity = '1';
      btn.style.cursor = 'pointer';
      btn.disabled = false;
    }
  } catch (err) {
    trainStatus.textContent = 'SYSTEM: ERROR LOADING MODEL.';
    console.error('MobileNet Error:', err);
    if (btn) {
      btn.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> NETWORK ERROR: COULD NOT LOAD AI';
      btn.style.cursor = 'not-allowed';
    }
  }
}

async function extractFeatures(src) {
  if (!mobileNetReady) throw new Error('MobileNet not ready');
  const embedding = tf.tidy(() => {
    const img = tf.browser.fromPixels(src).resizeBilinear([224, 224]);
    return mobileNet.infer(img.toFloat().div(127.5).sub(1).expandDims(0), true).squeeze();
  });
  const arr = await embedding.array();
  embedding.dispose();
  return arr;
}

function addPreview(cls, source) {
  if (!cls.previews) cls.previews = [];
  const tmp = document.createElement('canvas');
  tmp.width = 60; tmp.height = 60;
  tmp.getContext('2d').drawImage(source, 0, 0, 60, 60);
  cls.previews.push(tmp.toDataURL('image/jpeg', 0.5));
  if (cls.previews.length > 4) cls.previews.shift();
}

/* ══════════════════════════════════════════════════════════
   DASHBOARD RENDERING (OPTIMIZED)
 ══════════════════════════════════════════════════════════ */
function renderAll() {
  renderClassManager();
  renderWebcamButtons();
  updateStats();
  renderDistChart();
  renderThumbnails();
}

// Lightweight update for counts only to avoid destroying listeners
function updateCounts() {
  classes.forEach(cls => {
    const nodeStat = document.querySelector(`.node-stat[data-cid="${cls.id}"]`);
    if (nodeStat) nodeStat.textContent = cls.samples.length;
  });
  updateStats();
  renderDistChart();
}

function renderClassManager() {
  classManager.innerHTML = '';
  classes.forEach(cls => {
    const row = document.createElement('div');
    row.className = 'class-node';
    row.innerHTML = `
      <div class="node-color" style="background:${cls.color}"></div>
      <input class="node-input" type="text" value="${esc(cls.label)}" data-cid="${cls.id}">
      <span class="node-stat" data-cid="${cls.id}">${cls.samples.length}</span>
      <input type="file" id="fi_${cls.id}" accept="image/*,video/*,.gif" multiple hidden>
      <button class="node-upload" data-cid="${cls.id}" title="Upload Media"><i class="fa-solid fa-cloud-arrow-up"></i></button>
      <button class="node-del" data-cid="${cls.id}" title="Remove Class">✕</button>`;
    classManager.appendChild(row);
    
    row.querySelector('.node-upload').addEventListener('click', () => $('fi_' + cls.id).click());
    $('fi_' + cls.id).addEventListener('change', (e) => {
      processFilesForClass(e.target.files, cls);
      e.target.value = '';
    });
  });
  
  classManager.querySelectorAll('.node-input').forEach(inp => {
    inp.addEventListener('change', () => {
      const c = classes.find(x => x.id === +inp.dataset.cid);
      if (c) { c.label = inp.value.trim() || c.label; renderAll(); }
    });
  });
  classManager.querySelectorAll('.node-del').forEach(btn => {
    btn.addEventListener('click', () => {
      if (classes.length <= 2) { alert('System requires at least 2 classes.'); return; }
      classes = classes.filter(x => x.id !== +btn.dataset.cid);
      renderAll();
    });
  });
}

function renderWebcamButtons() {
  webcamRecBtns.innerHTML = '';
  classes.forEach((cls, i) => {
    const row = document.createElement('div');
    row.className = 'capture-row';

    const snapBtn = document.createElement('button');
    snapBtn.className = 'rec-pill';
    snapBtn.id = 'btn-snap-' + cls.id;
    snapBtn.innerHTML = `<span><i class="fa-solid fa-camera"></i> SNAP</span>`;
    snapBtn.addEventListener('click', () => takeSnapshot(cls.id));

    const recBtn = document.createElement('button');
    recBtn.className = 'rec-pill';
    recBtn.id = 'btn-rec-' + cls.id;
    recBtn.innerHTML = `<span><i class="fa-solid fa-video"></i> HOLD</span> <span class="k-tag">${i + 1}</span>`;
    
    const start = (e) => { e.preventDefault(); startRecording(cls.id); };
    const end = (e) => { e.preventDefault(); stopRecording(); };
    
    recBtn.addEventListener('mousedown', start);
    recBtn.addEventListener('mouseup', end);
    recBtn.addEventListener('mouseleave', end);
    recBtn.addEventListener('touchstart', start, { passive: false });
    recBtn.addEventListener('touchend', end);
    
    row.appendChild(snapBtn);
    row.appendChild(recBtn);
    webcamRecBtns.appendChild(row);
  });
}


function updateStats() {
  const total = classes.reduce((s,c) => s + c.samples.length, 0);
  if (statTotal) statTotal.textContent = total;
  if (statClasses) statClasses.textContent = classes.length;
  
  if (!total) { if(statBalance) statBalance.textContent = '0%'; return; }
  const counts = classes.map(c => c.samples.length);
  const mean = total / classes.length;
  const variance = counts.reduce((s,c) => s + (c-mean)**2, 0) / classes.length;
  const balance = Math.max(0, 100 - Math.round((Math.sqrt(variance) / (mean || 1)) * 100));
  if (statBalance) statBalance.textContent = balance + '%';
}

function renderDistChart() {
  const el = $('dist-chart');
  if (!el) return;
  const labels = classes.map(c => c.label);
  const data = classes.map(c => c.samples.length);
  const cols = classes.map(c => c.color);
  
  if (distChart) {
    distChart.data.labels = labels;
    distChart.data.datasets[0].data = data;
    distChart.data.datasets[0].backgroundColor = cols;
    distChart.update('none');
    return;
  }
  
  distChart = new Chart(el, {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: cols, borderRadius: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b949e', font: { family: 'Inter', size: 10 } }, grid: { display: false } },
        y: { ticks: { color: '#8b949e', font: { family: 'Inter', size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

function renderThumbnails() {
  if (!thumbGrid) return;
  thumbGrid.innerHTML = '';
  classes.forEach(cls => {
    (cls.previews || []).forEach(src => {
      const wrap = document.createElement('div');
      wrap.className = 'thumb-node';
      const img = new Image();
      img.src = src;
      img.style.width = '100%';
      img.style.height = '100%';
      img.style.objectFit = 'cover';
      wrap.appendChild(img);
      thumbGrid.appendChild(wrap);
    });
  });
}

/* ══════════════════════════════════════════════════════════
   WEBCAM ENGINE
 ══════════════════════════════════════════════════════════ */
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    // Use onloadeddata for better signal that video is actually flowing
    video.onloadeddata = () => {
      console.log('Video stream initialized');
    };
  } catch (e) { 
    console.warn('Webcam Access Denied:', e); 
  }
}

function drawLoop() {
  if (video.readyState >= 2) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  requestAnimationFrame(drawLoop);
}

function takeSnapshot(cid) {
  if (!mobileNetReady || isRecording) return;
  const cls = classes.find(c => c.id === cid);
  if (!cls || canvas.width === 0) return;
  
  const btn = $('btn-snap-' + cid);
  if(btn) {
    btn.classList.add('snap-pulse');
    setTimeout(() => btn.classList.remove('snap-pulse'), 400);
  }

  extractFeatures(canvas).then(arr => {
    cls.samples.push({ features: arr, imageData: null });
    addPreview(cls, canvas);
    updateCounts();
    renderThumbnails();
  }).catch(e => console.error('Snapshot Error:', e));
}

function startRecording(cid) {
  if (!mobileNetReady || isRecording) return;
  isRecording = true; recordingCId = cid;
  const btn = $('btn-rec-' + cid);
  if (btn) btn.classList.add('recording');
  
  // logP removed per user request

  recordInterval = setInterval(async () => {
    const cls = classes.find(c => c.id === cid);
    if (!cls) return;
    try {
      if (canvas.width === 0) return;
      
      const arr = await extractFeatures(canvas);
      cls.samples.push({ features: arr, imageData: null });
      addPreview(cls, canvas);
      
      // Update counts without full re-render
      updateCounts();
    } catch(e) { console.error('Recording Error:', e); }
  }, 100); // Faster sampling
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  clearInterval(recordInterval);
  if (recordingCId !== null) {
    const btn = $('btn-rec-' + recordingCId);
    if (btn) btn.classList.remove('recording');
    recordingCId = null;
    renderThumbnails();
  }
}

/* ══════════════════════════════════════════════════════════
   FILE PROCESSING
 ══════════════════════════════════════════════════════════ */
async function processFilesForClass(files, cls) {
  if (!files || !files.length) { logP('NO FILES SELECTED', 'err'); return; }
  
  let count = 0;
  for (const file of files) {
    try {
      if (file.type.startsWith('video') || file.type === 'image/gif') {
        await processVideo(file, cls);
      } else {
        await processImage(file, cls);
      }
      count++;
      
      // Memory/Freeze Anti-lock for massive datasets
      if (count % 25 === 0) {
        updateCounts(); // Give user feedback
        await new Promise(r => setTimeout(r, 0)); // Yield to main thread
      }
    } catch(e) { logP(`FAILED: ${file.name.toUpperCase()}`, 'err'); }
  }
  logP(`SUCCESSFULLY LOADED ${count} MEDIA SOURCE(S)`, 'ok');
  updateCounts();
  renderThumbnails();
}

async function processImage(file, cls) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = async () => {
      try {
        const tmp = document.createElement('canvas'); tmp.width = 224; tmp.height = 224;
        tmp.getContext('2d').drawImage(img, 0, 0, 224, 224);
        const arr = await extractFeatures(tmp);
        cls.samples.push({ features: arr, imageData: null });
        addPreview(cls, img);
        URL.revokeObjectURL(url); resolve();
      } catch(e) { reject(e); }
    };
    img.onerror = reject;
    img.src = url;
  });
}

async function processVideo(file, cls) {
  return new Promise((resolve, reject) => {
    const vid = document.createElement('video');
    vid.muted = true; vid.playsInline = true;
    const url = URL.createObjectURL(file);
    vid.src = url;
    vid.onloadedmetadata = async () => {
      try {
        const dur = vid.duration;
        const nFrames = Math.min(40, Math.max(10, Math.floor(dur * 8)));
        for (let i = 0; i < nFrames; i++) {
          vid.currentTime = (i / nFrames) * dur;
          await new Promise(r => vid.onseeked = r);
          const tmp = document.createElement('canvas'); tmp.width = 224; tmp.height = 224;
          tmp.getContext('2d').drawImage(vid, 0, 0, 224, 224);
          const arr = await extractFeatures(tmp);
          cls.samples.push({ features: arr, imageData: null });
          if(i % 5 === 0) addPreview(cls, vid);
        }
        URL.revokeObjectURL(url); resolve();
      } catch(e) { reject(e); }
    };
    vid.onerror = reject;
  });
}

/* ══════════════════════════════════════════════════════════
   TRAINING ENGINE
 ══════════════════════════════════════════════════════════ */
async function trainModel() {
  const minSamples = 5;
  const bad = classes.filter(c => c.samples.length < minSamples);
  if (bad.length) { 
    trainStatus.textContent = `SYSTEM: NEED SAMPLES (${minSamples}+) FOR ${bad[0].label.toUpperCase()}`; 
    logP(`ERROR: INSUFFICIENT DATA FOR ${bad[0].label.toUpperCase()}`, 'err');
    return; 
  }

  const nClasses = classes.length;
  const epochs = +cfgEpochs.value;
  const valSplit = +cfgValSplit.value;
  const lr = +cfgLR.value;

  btnTrain.disabled = true;
  btnTrain.classList.add('training-pulse');
  if(progressWrap) progressWrap.style.display = 'block';
  trainStatus.textContent = 'SYSTEM: OPTIMIZING NEURAL NETWORK...';
  logP('INITIALIZING TRAINING ENGINE...', 'sys');

  const allF = [], allL = [];
  classes.forEach((cls, idx) => cls.samples.forEach(s => { allF.push(s.features); allL.push(idx); }));
  
  const xs = tf.tensor2d(allF);
  const ys = tf.oneHot(tf.tensor1d(allL, 'int32'), nClasses);

  if (model) model.dispose();
  model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [xs.shape[1]], units: 512, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.5 }),
      tf.layers.dense({ units: 256, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({ units: nClasses, activation: 'softmax' })
    ]
  });

  model.compile({ optimizer: tf.train.adam(lr), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  initMetricsChart();

  logP('NETWORK COMPILED. MAXIMUM PARAMETERS LIVE.', 'sys');

  let bestValAcc = 0;
  let patienceCounter = 0;
  const PATIENCE = 20;

  await model.fit(xs, ys, {
    epochs, batchSize: 32, validationSplit: valSplit, shuffle: true,
    callbacks: {
      onEpochEnd: (ep, logs) => {
        const pct = Math.round(((ep + 1) / epochs) * 100);
        if(progressFill) progressFill.style.width = pct + '%';
        if(progressLabel) progressLabel.textContent = `TRAINING: EPOCH ${ep + 1}/${epochs}`;
        pushMetrics(ep + 1, logs.loss, logs.val_loss, logs.acc, logs.val_acc);

        // Advanced Early Stopping
        if (logs.val_acc > bestValAcc) {
          bestValAcc = logs.val_acc;
          patienceCounter = 0;
        } else {
          patienceCounter++;
          if (patienceCounter >= PATIENCE) {
            logP(`EARLY STOPPING TRIGGERED (No improvement for ${PATIENCE} epochs).`, 'sys');
            model.stopTraining = true;
          }
        }
      }
    }
  });

  trainStatus.textContent = 'SYSTEM: OPTIMIZATION COMPLETE.';
  logP('TRAINING SUCCESSFUL. MODEL LIVE.', 'ok');
  btnTrain.disabled = false;
  btnTrain.classList.remove('training-pulse');
  startLivePrediction();
}

function startLivePrediction() {
  if (predInterval) clearInterval(predInterval);
  predInterval = setInterval(async () => {
    if (!model || video.readyState < 2) return;
    tf.tidy(() => {
      const img = tf.browser.fromPixels(canvas).resizeBilinear([224, 224]);
      const feat = mobileNet.infer(img.toFloat().div(127.5).sub(1).expandDims(0), true);
      const pred = model.predict(feat);
      pred.data().then(probs => {
        const idx = probs.indexOf(Math.max(...probs));
        const conf = probs[idx];
        const cls = classes[idx];
        if (confFill) confFill.style.width = (conf * 100) + '%';
        if (confReadout) confReadout.textContent = (conf * 100).toFixed(1) + '%';
        if (confPrediction) confPrediction.textContent = `${cls.label.toUpperCase()} (${(conf * 100).toFixed(0)}%)`;
      });
    });
  }, 200);
}

/* ══════════════════════════════════════════════════════════
   UTILS & LISTENERS
 ══════════════════════════════════════════════════════════ */
function initListeners() {
  $('ob-start').addEventListener('click', () => {
    $('onboarding-overlay').style.opacity = '0';
    setTimeout(() => { $('onboarding-overlay').style.display = 'none'; }, 600);
  });

  cfgEpochs.addEventListener('input', () => { cfgEpochsVal.textContent = cfgEpochs.value; });
  cfgValSplit.addEventListener('input', () => { cfgValSplitVal.textContent = Math.round(+cfgValSplit.value * 100) + '%'; });

  btnTrain.addEventListener('click', trainModel);
  $('btn-add-class').addEventListener('click', () => {
    const id = nextClassId++;
    classes.push({ id, label: `class_${id+1}`, samples: [], previews: [], color: PALETTE[id % PALETTE.length] });
    renderAll();
  });

  $('btn-fullscreen').addEventListener('click', () => {
    const cc = document.querySelector('.camera-container');
    cc.classList.toggle('fullscreen');
    const icon = $('btn-fullscreen').querySelector('i');
    if (cc.classList.contains('fullscreen')) {
      icon.className = 'fa-solid fa-compress';
    } else {
      icon.className = 'fa-solid fa-expand';
    }
  });

  $('btn-save-model').addEventListener('click', async () => {
    if (!model) { alert('Train a model first.'); return; }
    // Instead of multiple files, we'll try to make it clearer
    trainStatus.textContent = 'SYSTEM: EXPORTING MODEL...';
    try {
      await model.save('downloads://custos-guardian-model');
      dlJSON({ 
        type: 'custos-meta',
        classes: classes.map(c => ({ id: c.id, label: c.label, color: c.color })) 
      }, 'custos-meta.json');
      trainStatus.textContent = 'SYSTEM: MODEL & META EXPORTED.';
    } catch(e) {
      trainStatus.textContent = 'SYSTEM: EXPORT FAILED.';
    }
  });

  $('btn-load-model').addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = '.json';
    input.onchange = async () => {
      try {
        if (model) model.dispose();
        model = await tf.loadLayersModel(tf.io.browserFiles([input.files[0], input.files[1]]));
        modelLoadStatus.textContent = 'SYSTEM: MODEL LOADED.';
        startLivePrediction();
      } catch(e) { 
        modelLoadStatus.textContent = 'SYSTEM: LOAD ERROR (Select both .json and .bin files).'; 
      }
    };
    input.setAttribute('multiple', '');
    input.click();
  });

  $('btn-export-json').addEventListener('click', () => {
    const data = { 
      type: 'custos-dataset',
      timestamp: new Date().toISOString(),
      classes: classes.map(c => ({ id: c.id, label: c.label, color: c.color, samples: c.samples.map(s => s.features) })) 
    };
    dlJSON(data, 'custos-dataset-export.json');
  });

  $('btn-import-data').addEventListener('click', () => $('import-data-input').click());
  $('import-data-input').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (data.classes) {
        classes = data.classes.map(c => ({ id: c.id, label: c.label, color: c.color, samples: (c.samples || []).map(f => ({ features: f, imageData: null })), previews: [] }));
        nextClassId = Math.max(...classes.map(c => c.id)) + 1;
        logP('DATASET IMPORTED SUCCESSFULLY (JSON)', 'ok');
        renderAll();
      }
    } catch(e) { logP('ERROR: INVALID DATA FORMAT', 'err'); }
  });

  $('btn-export-csv').addEventListener('click', exportCSV);
  $('btn-import-csv').addEventListener('click', () => $('import-csv-input').click());
  $('import-csv-input').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    try {
      const text = await file.text();
      importCSV(text);
      logP('DATASET IMPORTED SUCCESSFULLY (CSV/EXCEL)', 'ok');
      renderAll();
    } catch(e) { logP('ERROR: FAILED TO PARSE CSV', 'err'); }
  });

  $('btn-clear-data').addEventListener('click', () => {
    if (!confirm('Purge all collected samples and reset model?')) return;
    classes.forEach(c => c.samples = []);
    if (model) { model.dispose(); model = null; }
    if (predInterval) clearInterval(predInterval);
    if (metricsChart) { metricsChart.destroy(); metricsChart = null; initMetricsChart(); }
    if (distChart) { distChart.data.datasets[0].data = classes.map(c => 0); distChart.update(); }
    if (confFill) confFill.style.width = '0%';
    if (confPrediction) confPrediction.textContent = 'SYSTEM STANDBY';
    if (confReadout) confReadout.textContent = '0%';
    if (progressWrap) progressWrap.style.display = 'none';
    trainStatus.textContent = 'SYSTEM: DATA PURGED. START OVER.';
    logP('SYSTEM PURGED. ALL DATA ERASED.', 'sys');
    renderAll();
  });

  document.addEventListener('keydown', e => {
    if (isRecording) return;
    const idx = parseInt(e.key) - 1;
    if (idx >= 0 && idx < classes.length) startRecording(classes[idx].id);
  });
  document.addEventListener('keyup', e => {
    const idx = parseInt(e.key) - 1;
    if (idx >= 0 && idx < classes.length) stopRecording();
  });
}

function initMetricsChart() {
  const el = $('metrics-chart');
  if (!el) return;
  if (metricsChart) metricsChart.destroy();
  metricsChart = new Chart(el, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Loss', data: [], borderColor: '#ff3860', tension: 0.3, pointRadius: 0 },
      { label: 'Acc', data: [], borderColor: '#00ff88', tension: 0.3, pointRadius: 0 }
    ]},
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { ticks: { color: '#8b949e', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } } }
    }
  });
}

function pushMetrics(epoch, loss, valLoss, acc, valAcc) {
  if (!metricsChart) return;
  metricsChart.data.labels.push(epoch);
  metricsChart.data.datasets[0].data.push(loss);
  metricsChart.data.datasets[1].data.push(acc);
  metricsChart.update();
}

async function saveAsFile(content, defaultName, description, mimeType, extension) {
  try {
    if (window.showSaveFilePicker) {
      const handle = await window.showSaveFilePicker({
        suggestedName: defaultName,
        types: [{ description, accept: { [mimeType]: [extension] } }]
      });
      const writable = await handle.createWritable();
      await writable.write(content);
      await writable.close();
      return true;
    }
  } catch (e) {
    if (e.name === 'AbortError') return false; 
    console.warn('SavePicker Error:', e);
  }
  const blob = new Blob([content], { type: mimeType });
  const a = document.createElement('a'); 
  a.href = URL.createObjectURL(blob); a.download = defaultName; a.click();
  return true;
}

async function exportCSV() {
  let featureCount = 1280; // default for MobileNet v2
  for (let c of classes) {
    if (c.samples.length > 0) {
      featureCount = c.samples[0].features.length;
      break;
    }
  }
  
  let csv = 'ClassID,ClassLabel,' + Array.from({length: featureCount}, (_, i) => `Feature_${i}`).join(',') + '\n';
  classes.forEach(cls => {
    cls.samples.forEach(s => {
      csv += `${cls.id},${cls.label},${s.features.join(',')}\n`;
    });
  });
  
  const saved = await saveAsFile(csv, 'custos-dataset-export.csv', 'CSV File', 'text/csv', '.csv');
  if (saved) logP('DATASET EXPORTED AS CSV.', 'ok');
}

function importCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return;
  const newClasses = [];
  // Skip header
  for (let i = 1; i < lines.length; i++) {
    const row = lines[i].split(',');
    if (row.length < 130) continue;
    const cid = parseInt(row[0]);
    const label = row[1];
    const features = row.slice(2).map(Number);
    
    let cls = newClasses.find(c => c.id === cid);
    if (!cls) {
      cls = { id: cid, label, color: PALETTE[cid % PALETTE.length], samples: [] };
      newClasses.push(cls);
    }
    cls.samples.push({ features, imageData: null });
  }
  if (newClasses.length > 0) {
    classes = newClasses;
    nextClassId = Math.max(...classes.map(c => c.id)) + 1;
  }
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function logP(msg, cls) { const d = document.createElement('div'); d.className = cls; d.textContent = `> ${msg}`; procLog.appendChild(d); procLog.scrollTop = procLog.scrollHeight; }
async function dlJSON(obj, name) { 
  const jsonStr = JSON.stringify(obj, null, 2);
  await saveAsFile(jsonStr, name, 'JSON File', 'application/json', '.json');
}
(function init() {
  initListeners();
  loadMobileNet();
  startWebcam();
  drawLoop(); // FIX: Call drawLoop to start webcam canvas
  renderAll();
  logP('SYSTEM INITIALIZED. READY FOR INPUT.', 'sys');
})();

/*
 * 80–20 rule In AI POC development is not an oracle but a mirror! 
 * The 80–20 split has become a kind of ceremonial gesture in machine learning: 
 * take eighty percent of the data to teach the model, keep twenty percent aside to test it, 
 * and if the numbers sparkle, we congratulate ourselves
 */