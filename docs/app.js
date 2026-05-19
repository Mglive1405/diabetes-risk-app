/* ═══════════════ DiabetesIQ — Client-Side ML Inference ═══════════════ */

if (typeof ort !== 'undefined') {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = {
    "ort-wasm.wasm": "./ort-wasm.wasm",
    "ort-wasm-simd.wasm": "./ort-wasm-simd.wasm"
  };
}

// ─── State ──────────────────────────────────────────────────────────────
let session = null;
let currentGender = 'female';
let currentLang = 'en';
let lastResult = null; // Store last prediction for export

const TRANSLATIONS = {
  en: {
    nav_home: "Home", nav_assess: "Assess", nav_about: "About",
    hero_badge: "AI-Powered Health Tool", hero_title: "Know Your <span class=\"gradient-text\">Diabetes Risk</span> in Seconds",
    hero_subtitle: "Our machine learning model analyzes your health metrics to predict diabetes risk — completely private, running 100% in your browser.",
    stat_auc: "Mean CV AUC", stat_samples: "Training Samples", stat_private: "Private",
    hero_cta: "Start Assessment", form_title: "Health <span class=\"gradient-text\">Assessment</span>",
    form_subtitle: "Enter your medical information below. Fields marked with * are required. If you don't know a value, leave the default.",
    lbl_gender: "Gender", btn_female: "Female", btn_male: "Male",
    lbl_preg: "Pregnancies", hint_preg: "Number of times pregnant",
    lbl_gluc: "Glucose", hint_gluc: "Blood sugar level",
    lbl_bp: "Blood Pressure", hint_bp: "Diastolic blood pressure",
    lbl_skin: "Skin Thickness", hint_skin: "Triceps skin fold",
    lbl_ins: "Insulin", hint_ins: "2-hour serum insulin",
    lbl_dpf: "Diabetes Pedigree", hint_dpf: "Genetic likelihood score",
    lbl_age: "Age", hint_age: "Your current age",
    lbl_bmi: "BMI", hint_bmi: "Auto-calculated from height & weight",
    lbl_thresh: "Risk Threshold:", hint_thresh: "Adjust sensitivity — lower = more cautious",
    btn_loading: "Loading Model...", btn_analyze: "Analyze Risk",
    res_title: "Your <span class=\"gradient-text\">Results</span>",
    res_risk_label: "Risk", res_analyzing: "Analyzing...",
    res_factors_title: "Key Contributing Factors", res_factors_sub: "Your risk is heavily influenced by:",
    res_tips_title: "Personalized Health Tips",
    btn_download: "Download Report", btn_copy: "Copy Result", btn_new: "New Assessment",
    about_title: "About <span class=\"gradient-text\">DiabetesIQ</span>",
    about_card1_t: "AI-Powered", about_card1_d: "Logistic Regression model trained on 778 clinical records with proper cross-validation. Mean CV AUC: 0.83.",
    about_card2_t: "100% Private", about_card2_d: "All predictions run in your browser using ONNX Runtime. Your data never leaves your device.",
    about_card3_t: "Explainable", about_card3_d: "See which health metrics contribute most to your risk score with feature importance analysis.",
    about_card4_t: "Instant Results", about_card4_d: "Get your risk assessment in milliseconds. No server requests, no waiting, no sign-ups.",
    footer_disc_s: "Disclaimer:", footer_disc: "This tool is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare provider for medical concerns.",
    
    risk_high: "HIGH", risk_medium: "MEDIUM", risk_low: "LOW",
    exp_high: "We strongly recommend visiting a healthcare professional as soon as possible.",
    exp_medium: "You should pay attention to some health indicators and consult a doctor.",
    exp_low: "This suggests your current health indicators are within a safe range.",
    exp_prefix: "Your diabetes risk is",
    
    // Feature Labels
    feat_Pregnancies: 'Pregnancies', feat_Glucose: 'Glucose', feat_BloodPressure: 'Blood Pressure',
    feat_SkinThickness: 'Skin Thickness', feat_Insulin: 'Insulin', feat_BMI: 'BMI',
    feat_DiabetesPedigreeFunction: 'Diabetes Pedigree', feat_Age: 'Age'
  },
  ar: {
    nav_home: "الرئيسية", nav_assess: "التقييم", nav_about: "حول",
    hero_badge: "أداة صحية مدعومة بالذكاء الاصطناعي", hero_title: "اعرف <span class=\"gradient-text\">خطر السكري</span> في ثوانٍ",
    hero_subtitle: "يقوم نموذج التعلم الآلي الخاص بنا بتحليل مؤشراتك الصحية للتنبؤ بخطر الإصابة بالسكري - خاص تمامًا، يعمل 100% في متصفحك.",
    stat_auc: "متوسط المساحة تحت المنحنى", stat_samples: "عينات التدريب", stat_private: "الخصوصية",
    hero_cta: "ابدأ التقييم", form_title: "التقييم <span class=\"gradient-text\">الصحي</span>",
    form_subtitle: "أدخل معلوماتك الطبية أدناه. الحقول المميزة بعلامة * مطلوبة. إذا كنت لا تعرف قيمة، اترك الافتراضي.",
    lbl_gender: "الجنس", btn_female: "أنثى", btn_male: "ذكر",
    lbl_preg: "الحمل", hint_preg: "عدد مرات الحمل",
    lbl_gluc: "الجلوكوز", hint_gluc: "مستوى سكر الدم",
    lbl_bp: "ضغط الدم", hint_bp: "ضغط الدم الانبساطي",
    lbl_skin: "سمك الجلد", hint_skin: "طية جلد العضلة ثلاثية الرؤوس",
    lbl_ins: "الأنسولين", hint_ins: "أنسولين المصل لمدة ساعتين",
    lbl_dpf: "سجل السكري", hint_dpf: "درجة الاحتمالية الوراثية",
    lbl_age: "العمر", hint_age: "عمرك الحالي",
    lbl_bmi: "مؤشر كتلة الجسم", hint_bmi: "يُحسب تلقائياً من الطول والوزن",
    lbl_thresh: "عتبة الخطر:", hint_thresh: "تعديل الحساسية — أقل = أكثر حذراً",
    btn_loading: "جاري تحميل النموذج...", btn_analyze: "تحليل الخطر",
    res_title: "<span class=\"gradient-text\">نتائجك</span>",
    res_risk_label: "الخطر", res_analyzing: "جاري التحليل...",
    res_factors_title: "العوامل المساهمة الرئيسية", res_factors_sub: "يتأثر مستوى خطرك بشدة بـ:",
    res_tips_title: "نصائح صحية مخصصة",
    btn_download: "تحميل التقرير", btn_copy: "نسخ النتيجة", btn_new: "تقييم جديد",
    about_title: "حول <span class=\"gradient-text\">DiabetesIQ</span>",
    about_card1_t: "ذكاء اصطناعي", about_card1_d: "نموذج انحدار لوجستي مدرب على 778 سجلاً سريرياً بمتوسط AUC يبلغ 0.83.",
    about_card2_t: "خاص 100%", about_card2_d: "تعمل جميع التنبؤات في متصفحك. لا تترك بياناتك جهازك أبدًا.",
    about_card3_t: "قابل للتفسير", about_card3_d: "تعرف على المقاييس الصحية التي تساهم أكثر في درجة الخطر الخاصة بك.",
    about_card4_t: "نتائج فورية", about_card4_d: "احصل على تقييمك في أجزاء من الثانية. بدون طلبات خادم أو انتظار.",
    footer_disc_s: "إخلاء مسؤولية:", footer_disc: "هذه الأداة للأغراض التعليمية فقط وليست بديلاً عن الاستشارة الطبية المهنية. استشر دائمًا مقدم رعاية صحية لمخاوفك الطبية.",
    
    risk_high: "مرتفع", risk_medium: "متوسط", risk_low: "منخفض",
    exp_high: "نوصي بشدة بزيارة أخصائي رعاية صحية في أقرب وقت ممكن.",
    exp_medium: "يجب عليك الانتباه لبعض المؤشرات الصحية واستشارة طبيب.",
    exp_low: "هذا يشير إلى أن مؤشراتك الصحية الحالية تقع ضمن النطاق الآمن.",
    exp_prefix: "خطر إصابتك بمرض السكري",
    
    feat_Pregnancies: 'الحمل', feat_Glucose: 'الجلوكوز', feat_BloodPressure: 'ضغط الدم',
    feat_SkinThickness: 'سمك الجلد', feat_Insulin: 'الأنسولين', feat_BMI: 'كتلة الجسم',
    feat_DiabetesPedigreeFunction: 'الوراثة', feat_Age: 'العمر'
  }
};

const FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'];
const AVERAGES = { Pregnancies: 3.8, Glucose: 121.7, BloodPressure: 72.4, SkinThickness: 29.1, Insulin: 140.7, BMI: 32.5, DiabetesPedigreeFunction: 0.47, Age: 33.2 };

const HEALTH_TIPS = {
  Glucose: { icon: '🍭', en: { title: 'Blood Sugar', tip: 'Your glucose level is a key factor. Reduce refined carbs and sugary drinks. Opt for whole grains.' }, ar: { title: 'سكر الدم', tip: 'مستوى الجلوكوز لديك عامل رئيسي. قلل من الكربوهيدرات المكررة والمشروبات السكرية.' } },
  BMI: { icon: '🏃', en: { title: 'Body Weight', tip: 'BMI is significantly impacting your score. Aim for 30+ minutes of moderate exercise daily.' }, ar: { title: 'وزن الجسم', tip: 'مؤشر كتلة الجسم يؤثر بشكل كبير. اهدف إلى 30+ دقيقة من التمارين المعتدلة يوميًا.' } },
  Age: { icon: '⏰', en: { title: 'Age Factor', tip: 'Age increases risk naturally. Stay proactive with annual screenings.' }, ar: { title: 'عامل العمر', tip: 'يزيد العمر من الخطر بشكل طبيعي. ابق استباقياً مع الفحوصات السنوية.' } },
  Insulin: { icon: '💉', en: { title: 'Insulin Levels', tip: 'Regular checkups help monitor insulin resistance. Consider reducing processed foods.' }, ar: { title: 'مستويات الأنسولين', tip: 'تساعد الفحوصات المنتظمة في مراقبة مقاومة الأنسولين.' } },
  Pregnancies: { icon: '👶', en: { title: 'Pregnancy History', tip: 'Multiple pregnancies can affect glucose metabolism. Monitor blood sugar closely.' }, ar: { title: 'تاريخ الحمل', tip: 'يمكن أن يؤثر تكرار الحمل على استقلاب الجلوكوز.' } },
  BloodPressure: { icon: '❤️', en: { title: 'Blood Pressure', tip: 'Reduce salt intake, manage stress, and exercise regularly.' }, ar: { title: 'ضغط الدم', tip: 'قلل من تناول الملح، أدر التوتر، ومارس الرياضة بانتظام.' } },
  SkinThickness: { icon: '💪', en: { title: 'Body Composition', tip: 'Skin thickness relates to body fat distribution. Stay active.' }, ar: { title: 'تكوين الجسم', tip: 'يرتبط سمك الجلد بتوزيع دهون الجسم. حافظ على نشاطك.' } },
  DiabetesPedigreeFunction: { icon: '🧬', en: { title: 'Genetic Risk', tip: 'Family history plays a role. Be extra vigilant with a healthy lifestyle.' }, ar: { title: 'الخطر الوراثي', tip: 'يلعب التاريخ العائلي دوراً. كن أكثر يقظة مع أسلوب حياة صحي.' } }
};

// ─── Localization ───────────────────────────────────────────────────────
function updateLanguageUI() {
  const t = TRANSLATIONS[currentLang];
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (t[key]) el.innerHTML = t[key];
  });
  
  const btnText = document.getElementById('btn-text');
  if (btnText && !document.getElementById('submit-btn').disabled) {
    btnText.textContent = t.btn_analyze;
  }
}

function switchLanguage() {
  currentLang = currentLang === 'en' ? 'ar' : 'en';
  document.documentElement.setAttribute('dir', currentLang === 'ar' ? 'rtl' : 'ltr');
  document.documentElement.lang = currentLang;
  document.getElementById('lang-toggle').textContent = currentLang === 'en' ? 'العربية' : 'English';
  updateLanguageUI();
  
  if (lastResult) {
    renderGauge(lastResult.prob, lastResult.riskLevel);
    renderImportance(lastResult.importances);
    renderTips(lastResult.importances, lastResult.riskLevel);
  }
}

// ─── Initialize ONNX ────────────────────────────────────────────────────
async function initModel() {
  console.log('Model loading started...');
  try {
    session = await ort.InferenceSession.create('./diabetes_model.onnx');
    console.log('ONNX model loaded successfully');
    
    const errorEl = document.getElementById('model-error');
    if (errorEl) errorEl.classList.add('hidden');
    
    const btn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    if (btn) {
      btn.disabled = false;
      btn.classList.remove('loading');
    }
    if (btnText) {
      btnText.textContent = TRANSLATIONS[currentLang].btn_analyze;
    }
  } catch (e) {
    console.error('Failed to load ONNX model:', e);
    const errorEl = document.getElementById('model-error');
    errorEl.textContent = `Error loading AI model: ${e.message || e}. Please check console.`;
    errorEl.classList.remove('hidden');
  }
}

// ─── Predict ────────────────────────────────────────────────────────────
async function predict(inputValues) {
  if (!session) { await initModel(); }
  const tensor = new ort.Tensor('float32', Float32Array.from(inputValues), [1, 8]);
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);
  return results.probabilities.data[0];
}

function updateBMI() {
  const h = parseFloat(document.getElementById('height').value) || 165;
  const w = parseFloat(document.getElementById('weight').value) || 70;
  const bmi = (w / ((h / 100) ** 2)).toFixed(1);
  document.getElementById('bmi-value').textContent = bmi;
  return parseFloat(bmi);
}

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

function setupThreshold() {
  const slider = document.getElementById('threshold');
  const display = document.getElementById('threshold-value');
  slider.addEventListener('input', () => {
    display.textContent = slider.value + '%';
  });
}

function computeImportance(values) {
  const importances = {};
  FEATURES.forEach((f, i) => {
    const deviation = Math.abs(values[i] - AVERAGES[f]) / (AVERAGES[f] || 1);
    importances[f] = deviation;
  });
  const max = Math.max(...Object.values(importances), 0.01);
  Object.keys(importances).forEach(k => { importances[k] = importances[k] / max; });
  return importances;
}

function renderGauge(prob, riskLevel) {
  const t = TRANSLATIONS[currentLang];
  const pct = (prob * 100).toFixed(1);
  const gaugeEl = document.getElementById('gauge-fill');
  const percentEl = document.getElementById('gauge-percent');
  const badgeEl = document.getElementById('risk-badge');
  const expEl = document.getElementById('risk-explanation');

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
    const grad = svg.querySelector('#gaugeGradient');
    const colors = riskLevel === 'High' ? ['#ef4444','#f97316'] : riskLevel === 'Medium' ? ['#f59e0b','#eab308'] : ['#10b981','#06b6d4'];
    const stops = grad.querySelectorAll('stop');
    stops[0].setAttribute('stop-color', colors[0]);
    stops[1].setAttribute('stop-color', colors[1]);
  }

  const circumference = 2 * Math.PI * 85; 
  const offset = circumference * (1 - prob);
  gaugeEl.style.strokeDashoffset = offset;

  animateCounter(percentEl, prob * 100);

  const riskKey = riskLevel === 'High' ? 'risk_high' : riskLevel === 'Medium' ? 'risk_medium' : 'risk_low';
  const expKey = riskLevel === 'High' ? 'exp_high' : riskLevel === 'Medium' ? 'exp_medium' : 'exp_low';

  percentEl.style.color = riskLevel === 'High' ? '#ef4444' : riskLevel === 'Medium' ? '#f59e0b' : '#10b981';
  
  badgeEl.innerHTML = `${t.exp_prefix} <strong style="color: inherit;">${t[riskKey]}</strong>`;
  badgeEl.className = 'risk-badge risk-' + riskLevel.toLowerCase();
  
  expEl.innerHTML = `${t[expKey]} <br><br> <span style="font-size: 0.85em; opacity: 0.8;">(${pct}%)</span>`;
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

function renderImportance(importances) {
  const t = TRANSLATIONS[currentLang];
  const container = document.getElementById('importance-bars');
  container.innerHTML = '';
  const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 6);

  sorted.forEach(([feat, val], i) => {
    const row = document.createElement('div');
    row.className = 'imp-row';
    const label = t['feat_' + feat];
    row.innerHTML = `
      <span class="imp-label">${label}</span>
      <div class="imp-bar-bg"><div class="imp-bar-fill" style="width: 0%;" id="bar-${i}"></div></div>
      <span class="imp-val">${(val * 100).toFixed(0)}%</span>
    `;
    container.appendChild(row);

    requestAnimationFrame(() => {
      setTimeout(() => {
        const bar = document.getElementById('bar-' + i);
        if (bar) bar.style.width = (val * 100).toFixed(0) + '%';
      }, 100 + i * 80);
    });
  });
}

function renderTips(importances, riskLevel) {
  const container = document.getElementById('tips-list');
  container.innerHTML = '';
  const topFactors = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 3);

  if (riskLevel === 'Low') {
    const item = document.createElement('div');
    item.className = 'tip-item';
    const msg = currentLang === 'ar' ? '<strong>أداء ممتاز!</strong> عوامل الخطر ضمن النطاقات الصحية. استمر في نمط حياتك الصحي.' : '<strong>Looking Good!</strong> Your risk factors are within healthy ranges. Keep up your healthy lifestyle.';
    item.innerHTML = `<span class="tip-icon">✅</span><span class="tip-text">${msg}</span>`;
    container.appendChild(item);
  }

  topFactors.forEach(([feat]) => {
    const tipData = HEALTH_TIPS[feat];
    if (!tipData) return;
    const locTip = tipData[currentLang];
    const item = document.createElement('div');
    item.className = 'tip-item';
    item.innerHTML = `
      <span class="tip-icon">${tipData.icon}</span>
      <span class="tip-text"><strong>${locTip.title}:</strong> ${locTip.tip}</span>
    `;
    container.appendChild(item);
  });
}

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

    lastResult = { prob, riskLevel, importances };

    const resultsEl = document.getElementById('results');
    resultsEl.classList.remove('hidden');

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

function resetForm() {
  document.getElementById('results').classList.add('hidden');
  document.getElementById('gauge-fill').style.strokeDashoffset = 534;
  document.getElementById('gauge-percent').textContent = '0%';
  document.getElementById('assess').scrollIntoView({ behavior: 'smooth' });
}

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

function copyToClipboard() {
  if (!lastResult) return;
  const t = TRANSLATIONS[currentLang];
  const riskKey = lastResult.riskLevel === 'High' ? 'risk_high' : lastResult.riskLevel === 'Medium' ? 'risk_medium' : 'risk_low';
  const pct = (lastResult.prob * 100).toFixed(1);
  const text = currentLang === 'en' 
    ? `My diabetes risk is ${t[riskKey]} (${pct}%) using DiabetesIQ. Check your risk securely here!`
    : `خطر إصابتي بالسكري ${t[riskKey]} (${pct}%) باستخدام DiabetesIQ. تحقق من مستوى خطرك هنا!`;
    
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('btn-copy');
    const orig = btn.innerHTML;
    btn.innerHTML = `<span>${currentLang === 'en' ? 'Copied!' : 'تم النسخ!'}</span>`;
    setTimeout(() => btn.innerHTML = orig, 2000);
  });
}

function generatePDF() {
  if (!lastResult || !window.jspdf) return;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  
  const pct = (lastResult.prob * 100).toFixed(1);
  
  doc.setFont("helvetica", "bold");
  doc.setFontSize(22);
  doc.setTextColor(6, 182, 212);
  doc.text("DiabetesIQ - Health Assessment", 20, 30);
  
  doc.setFontSize(14);
  doc.setTextColor(100, 100, 100);
  doc.text("AI-Powered Risk Prediction Report", 20, 40);
  
  doc.setFont("helvetica", "normal");
  doc.setTextColor(0, 0, 0);
  doc.setFontSize(16);
  doc.text(`Overall Risk: ${lastResult.riskLevel.toUpperCase()} (${pct}%)`, 20, 60);
  
  doc.setFontSize(12);
  doc.text("Top Contributing Factors:", 20, 80);
  
  const sorted = Object.entries(lastResult.importances).sort((a, b) => b[1] - a[1]).slice(0, 4);
  let y = 90;
  sorted.forEach(([feat, val]) => {
    doc.text(`- ${feat}: Impact ${(val*100).toFixed(0)}%`, 25, y);
    y += 10;
  });
  
  y += 10;
  doc.setFont("helvetica", "bold");
  doc.text("Recommendations:", 20, y);
  doc.setFont("helvetica", "normal");
  y += 10;
  
  const topFactors = sorted.slice(0, 2);
  topFactors.forEach(([feat]) => {
    const tipData = HEALTH_TIPS[feat];
    if (tipData) {
      const tipText = doc.splitTextToSize(`- ${tipData.en.title}: ${tipData.en.tip}`, 170);
      doc.text(tipText, 25, y);
      y += 7 * tipText.length;
    }
  });

  doc.save("DiabetesIQ-Report.pdf");
}

document.addEventListener('DOMContentLoaded', () => {
  initModel();
  setupGenderToggle();
  setupThreshold();
  setupNavbar();
  updateBMI();

  document.getElementById('height').addEventListener('input', updateBMI);
  document.getElementById('weight').addEventListener('input', updateBMI);
  document.getElementById('prediction-form').addEventListener('submit', handleSubmit);
  document.getElementById('lang-toggle').addEventListener('click', switchLanguage);
});
