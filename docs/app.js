/* ═══════════════ DiabetesIQ — Client-Side ML Inference ═══════════════ */

// ─── State ──────────────────────────────────────────────────────────────
let session = null;
let currentGender = 'female';

// ─── Feature metadata ──────────────────────────────────────────────────
const FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'];

const FEATURE_LABELS = {
  Pregnancies: 'Pregnancies',
  Glucose: 'Glucose',
  BloodPressure: 'Blood Pressure',
  SkinThickness: 'Skin Thickness',
  Insulin: 'Insulin',
  BMI: 'BMI',
  DiabetesPedigreeFunction: 'Diabetes Pedigree',
  Age: 'Age'
};

// Average values used for "importance" baseline comparison
const AVERAGES = {
  Pregnancies: 3.8, Glucose: 121.7, BloodPressure: 72.4,
  SkinThickness: 29.1, Insulin: 140.7, BMI: 32.5,
  DiabetesPedigreeFunction: 0.47, Age: 33.2
};

const HEALTH_TIPS = {
  Glucose: { icon: '🍭', title: 'Blood Sugar', tip: 'Your glucose level is a key factor. Reduce refined carbs and sugary drinks. Opt for whole grains, fiber-rich foods, and regular meals.' },
  BMI: { icon: '🏃', title: 'Body Weight', tip: 'BMI is significantly impacting your score. Aim for 30+ minutes of moderate exercise daily and consult a dietitian for a personalized plan.' },
  Age: { icon: '⏰', title: 'Age Factor', tip: 'Age increases risk naturally. Stay proactive with annual screenings, especially after 35. Early detection is key.' },
  Insulin: { icon: '💉', title: 'Insulin Levels', tip: 'Your insulin level matters. Regular checkups help monitor insulin resistance. Consider reducing processed foods.' },
  Pregnancies: { icon: '👶', title: 'Pregnancy History', tip: 'Multiple pregnancies can affect glucose metabolism. Monitor blood sugar closely post-pregnancy.' },
  BloodPressure: { icon: '❤️', title: 'Blood Pressure', tip: 'Blood pressure contributes to your risk. Reduce salt intake, manage stress, and exercise regularly.' },
  SkinThickness: { icon: '💪', title: 'Body Composition', tip: 'Skin thickness relates to body fat distribution. Stay active and maintain a balanced diet.' },
  DiabetesPedigreeFunction: { icon: '🧬', title: 'Genetic Risk', tip: 'Family history plays a role. Be extra vigilant with regular screenings and a healthy lifestyle.' }
};

// ─── Initialize ONNX ────────────────────────────────────────────────────
async function initModel() {
  try {
    // Configure ONNX Runtime Web WASM options
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
    
    session = await ort.InferenceSession.create('./diabetes_model.onnx');
    console.log('ONNX model loaded successfully');
  } catch (e) {
    console.error('Failed to load ONNX model:', e);
    alert('Failed to load AI model. Please refresh the page.');
  }
}

// ─── Predict ────────────────────────────────────────────────────────────
async function predict(inputValues) {
  if (!session) { await initModel(); }
  const tensor = new ort.Tensor('float32', Float32Array.from(inputValues), [1, 8]);
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);

  // Output: probabilities tensor [1, 2] — class 0 and class 1
  const probOutput = results[session.outputNames[1]];
  const prob = probOutput.data[1]; // probability of diabetes (class 1)
  return prob;
}

// ─── BMI Calculation ────────────────────────────────────────────────────
function updateBMI() {
  const h = parseFloat(document.getElementById('height').value) || 165;
  const w = parseFloat(document.getElementById('weight').value) || 70;
  const bmi = (w / ((h / 100) ** 2)).toFixed(1);
  document.getElementById('bmi-value').textContent = bmi;
  return parseFloat(bmi);
}

// ─── Gender Toggle ──────────────────────────────────────────────────────
function setupGenderToggle() {
  const btns = document.querySelectorAll('.gender-btn');
  const pregGroup = document.getElementById('pregnancies-group');

  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentGender = btn.dataset.gender;
      if (currentGender === 'male') {
        pregGroup.style.display = 'none';
        document.getElementById('pregnancies').value = 0;
      } else {
        pregGroup.style.display = '';
      }
    });
  });
}

// ─── Threshold Display ──────────────────────────────────────────────────
function setupThreshold() {
  const slider = document.getElementById('threshold');
  const display = document.getElementById('threshold-value');
  slider.addEventListener('input', () => {
    display.textContent = slider.value + '%';
  });
}

// ─── Compute simple feature importance (deviation from average) ────────
function computeImportance(values) {
  const importances = {};
  FEATURES.forEach((f, i) => {
    const deviation = Math.abs(values[i] - AVERAGES[f]) / (AVERAGES[f] || 1);
    importances[f] = deviation;
  });
  // Normalize
  const max = Math.max(...Object.values(importances), 0.01);
  Object.keys(importances).forEach(k => { importances[k] = importances[k] / max; });
  return importances;
}

// ─── Render Gauge ───────────────────────────────────────────────────────
function renderGauge(prob, riskLevel) {
  const pct = (prob * 100).toFixed(1);
  const gaugeEl = document.getElementById('gauge-fill');
  const percentEl = document.getElementById('gauge-percent');
  const badgeEl = document.getElementById('risk-badge');

  // SVG gradient injection (only once)
  const svg = document.getElementById('gauge-svg');
  if (!svg.querySelector('defs')) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    grad.id = 'gaugeGradient';
    const colors = riskLevel === 'High' ? ['#ef4444','#f97316'] : riskLevel === 'Medium' ? ['#f59e0b','#eab308'] : ['#10b981','#06b6d4'];
    const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    s1.setAttribute('offset', '0%'); s1.setAttribute('stop-color', colors[0]);
    const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    s2.setAttribute('offset', '100%'); s2.setAttribute('stop-color', colors[1]);
    grad.appendChild(s1); grad.appendChild(s2);
    defs.appendChild(grad); svg.insertBefore(defs, svg.firstChild);
  } else {
    // Update gradient colors
    const grad = svg.querySelector('#gaugeGradient');
    const colors = riskLevel === 'High' ? ['#ef4444','#f97316'] : riskLevel === 'Medium' ? ['#f59e0b','#eab308'] : ['#10b981','#06b6d4'];
    const stops = grad.querySelectorAll('stop');
    stops[0].setAttribute('stop-color', colors[0]);
    stops[1].setAttribute('stop-color', colors[1]);
  }

  const circumference = 2 * Math.PI * 85; // ~534
  const offset = circumference * (1 - prob);
  gaugeEl.style.strokeDashoffset = offset;

  // Animate counter
  animateCounter(percentEl, prob * 100);

  // Risk badge
  percentEl.style.color = riskLevel === 'High' ? '#ef4444' : riskLevel === 'Medium' ? '#f59e0b' : '#10b981';
  badgeEl.textContent = riskLevel + ' Risk';
  badgeEl.className = 'risk-badge risk-' + riskLevel.toLowerCase();
}

function animateCounter(el, target) {
  let current = 0;
  const duration = 1500;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    current = eased * target;
    el.textContent = current.toFixed(1) + '%';
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ─── Render Feature Importance Bars ─────────────────────────────────────
function renderImportance(importances) {
  const container = document.getElementById('importance-bars');
  container.innerHTML = '';

  const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 6);

  sorted.forEach(([feat, val], i) => {
    const row = document.createElement('div');
    row.className = 'imp-row';
    row.innerHTML = `
      <span class="imp-label">${FEATURE_LABELS[feat]}</span>
      <div class="imp-bar-bg"><div class="imp-bar-fill" style="width: 0%;" id="bar-${i}"></div></div>
      <span class="imp-val">${(val * 100).toFixed(0)}%</span>
    `;
    container.appendChild(row);

    // Animate after append
    requestAnimationFrame(() => {
      setTimeout(() => {
        document.getElementById('bar-' + i).style.width = (val * 100).toFixed(0) + '%';
      }, 100 + i * 80);
    });
  });
}

// ─── Render Health Tips ─────────────────────────────────────────────────
function renderTips(importances, riskLevel) {
  const container = document.getElementById('tips-list');
  container.innerHTML = '';

  const topFactors = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 3);

  if (riskLevel === 'Low') {
    const item = document.createElement('div');
    item.className = 'tip-item';
    item.innerHTML = `
      <span class="tip-icon">✅</span>
      <span class="tip-text"><strong>Looking Good!</strong> Your risk factors are within healthy ranges. Keep up your healthy lifestyle with regular exercise and balanced nutrition.</span>
    `;
    container.appendChild(item);
  }

  topFactors.forEach(([feat]) => {
    const tipData = HEALTH_TIPS[feat];
    if (!tipData) return;
    const item = document.createElement('div');
    item.className = 'tip-item';
    item.innerHTML = `
      <span class="tip-icon">${tipData.icon}</span>
      <span class="tip-text"><strong>${tipData.title}:</strong> ${tipData.tip}</span>
    `;
    container.appendChild(item);
  });
}

// ─── Form Submit ────────────────────────────────────────────────────────
async function handleSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  btn.classList.add('loading');

  const bmi = updateBMI();
  const threshold = parseInt(document.getElementById('threshold').value) / 100;

  const values = [
    parseFloat(document.getElementById('pregnancies').value) || 0,
    parseFloat(document.getElementById('glucose').value),
    parseFloat(document.getElementById('blood-pressure').value),
    parseFloat(document.getElementById('skin-thickness').value),
    parseFloat(document.getElementById('insulin').value),
    bmi,
    parseFloat(document.getElementById('dpf').value),
    parseFloat(document.getElementById('age').value)
  ];

  try {
    const prob = await predict(values);
    const riskLevel = prob >= threshold ? 'High' : prob >= 0.35 ? 'Medium' : 'Low';

    const importances = computeImportance(values);

    // Show results
    const resultsEl = document.getElementById('results');
    resultsEl.classList.remove('hidden');

    // Scroll to results
    setTimeout(() => {
      resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    renderGauge(prob, riskLevel);
    renderImportance(importances);
    renderTips(importances, riskLevel);

  } catch (err) {
    console.error('Prediction failed:', err);
    alert('Prediction failed. Please check your inputs and try again.');
  } finally {
    setTimeout(() => { btn.classList.remove('loading'); }, 600);
  }
}

// ─── Reset ──────────────────────────────────────────────────────────────
function resetForm() {
  document.getElementById('results').classList.add('hidden');
  document.getElementById('gauge-fill').style.strokeDashoffset = 534;
  document.getElementById('gauge-percent').textContent = '0%';
  document.getElementById('assess').scrollIntoView({ behavior: 'smooth' });
}

// ─── Navbar Scroll Effect ───────────────────────────────────────────────
function setupNavbar() {
  const nav = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      nav.style.borderBottomColor = 'rgba(255,255,255,0.1)';
      nav.style.background = 'rgba(10, 14, 23, 0.95)';
    } else {
      nav.style.borderBottomColor = 'rgba(255,255,255,0.04)';
      nav.style.background = 'rgba(10, 14, 23, 0.8)';
    }
  });
}

// ─── Init ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initModel();
  setupGenderToggle();
  setupThreshold();
  setupNavbar();
  updateBMI();

  // BMI auto-update
  document.getElementById('height').addEventListener('input', updateBMI);
  document.getElementById('weight').addEventListener('input', updateBMI);

  // Form submit
  document.getElementById('prediction-form').addEventListener('submit', handleSubmit);
});
