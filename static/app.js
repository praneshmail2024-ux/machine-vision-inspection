/* ═══════════════════════════════════════════════════════════════════════════
   Mission Vision – Premium Dashboard Controller
   ═══════════════════════════════════════════════════════════════════════════ */

const feed = document.querySelector('#cameraFeed');
const debugFeed = document.querySelector('#debugFeed');
const statusMessage = document.querySelector('#statusMessage');
const badge = document.querySelector('#liveBadge');
const menuButton = document.querySelector('#menuButton');
const closeDrawerButton = document.querySelector('#closeDrawerButton');
const settingsDrawer = document.querySelector('#settingsDrawer');
const drawerScrim = document.querySelector('#drawerScrim');
const cameraStage = document.querySelector('.camera-stage');
const fullscreenButton = document.querySelector('#fullscreenButton');
const currentTime = document.querySelector('#currentTime');
const toastContainer = document.querySelector('#toastContainer');
const sliderIds = ['left', 'right', 'top', 'bottom', 'countingLine'];

/* ── Subtle Particle Background ─────────────────────────────────────────── */
(function initParticles() {
  const canvas = document.querySelector('#particleCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, particles = [];
  const COUNT = 40;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  function createParticle() {
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5,
      alpha: Math.random() * 0.3 + 0.05,
    };
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 221, 179, ${p.alpha})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  resize();
  for (let i = 0; i < COUNT; i++) particles.push(createParticle());
  window.addEventListener('resize', resize);
  draw();
})();

/* ── Toast Notification System ──────────────────────────────────────────── */
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    if (toast.parentNode) toast.remove();
  }, 4000);
}

/* ── Drawer Management ──────────────────────────────────────────────────── */
function openDrawer() {
  drawerScrim.hidden = false;
  requestAnimationFrame(() => drawerScrim.classList.add('visible'));
  settingsDrawer.classList.add('open');
  settingsDrawer.setAttribute('aria-hidden', 'false');
  menuButton.setAttribute('aria-expanded', 'true');
  closeDrawerButton.focus();
  refreshDebugFrame();
}

function closeDrawer() {
  drawerScrim.classList.remove('visible');
  settingsDrawer.classList.remove('open');
  settingsDrawer.setAttribute('aria-hidden', 'true');
  menuButton.setAttribute('aria-expanded', 'false');
  window.setTimeout(() => {
    if (!settingsDrawer.classList.contains('open')) drawerScrim.hidden = true;
  }, 400);
  menuButton.focus();
}

function selectDrawerTab(selectedTab) {
  const tabs = [
    { button: document.querySelector('#pipelineTab'), panel: document.querySelector('#pipelinePanel') },
    { button: document.querySelector('#fovTab'), panel: document.querySelector('#fovPanel') },
  ];
  tabs.forEach(({ button, panel }) => {
    const active = button === selectedTab;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
    panel.hidden = !active;
  });
  if (selectedTab.id === 'pipelineTab') refreshDebugFrame();
}

function openDrawerAt(selectedTab) {
  selectDrawerTab(selectedTab);
  openDrawer();
}

function setActiveNavigation(selected) {
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item === selected));
}

menuButton.addEventListener('click', openDrawer);
closeDrawerButton.addEventListener('click', closeDrawer);
drawerScrim.addEventListener('click', closeDrawer);
document.querySelector('#pipelineTab').addEventListener('click', (event) => selectDrawerTab(event.currentTarget));
document.querySelector('#fovTab').addEventListener('click', (event) => selectDrawerTab(event.currentTarget));
document.querySelector('[data-view="live"]').addEventListener('click', (event) => {
  setActiveNavigation(event.currentTarget);
  closeDrawer();
});
document.querySelector('#pipelineNavButton').addEventListener('click', (event) => {
  setActiveNavigation(event.currentTarget);
  openDrawerAt(document.querySelector('#pipelineTab'));
});
document.querySelector('#fovNavButton').addEventListener('click', (event) => {
  setActiveNavigation(event.currentTarget);
  openDrawerAt(document.querySelector('#fovTab'));
});
document.querySelector('#calibrationNavButton').addEventListener('click', (event) => {
  setActiveNavigation(event.currentTarget);
  openDrawerAt(document.querySelector('#fovTab'));
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && settingsDrawer.classList.contains('open')) closeDrawer();
});

/* ── Fullscreen Toggle ──────────────────────────────────────────────────── */
fullscreenButton.addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await cameraStage.requestFullscreen();
  } catch {
    showToast('Fullscreen is not available in this browser.', 'error');
  }
});

/* ── Clock ──────────────────────────────────────────────────────────────── */
function updateClock() {
  currentTime.textContent = new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true,
  }).format(new Date());
}

/* ── Frame Refresh ──────────────────────────────────────────────────────── */
function refreshFrame() {
  // The backend serves an OpenCV JPEG snapshot. A fixed timer is more reliable
  // than chaining image-load events, which can be missed when the page opens.
  feed.src = `/api/frame.jpg?t=${Date.now()}`;
}

function refreshDebugFrame() {
  const pipelineVisible = settingsDrawer.classList.contains('open') && !document.querySelector('#pipelinePanel').hidden;
  if (pipelineVisible) debugFeed.src = `/api/debug.jpg?t=${Date.now()}`;
}

/* ── Status & Metrics ───────────────────────────────────────────────────── */
function showStatus(message, running) {
  statusMessage.textContent = message;
  badge.replaceChildren();
  const dot = document.createElement('i');
  badge.append(dot, running ? 'LIVE' : 'OFFLINE');
  badge.classList.toggle('active', running);
}

/* Animated counter bump effect */
let prevTotal = -1;
let prevVisible = -1;

function animateCount(element, newValue) {
  element.textContent = newValue;
  element.classList.remove('count-bump');
  // Force reflow to restart animation
  void element.offsetWidth;
  element.classList.add('count-bump');
}

function showMetrics(data) {
  const visibleEl = document.querySelector('#visibleCount');
  const totalEl = document.querySelector('#totalCount');
  const newVisible = data.visible ?? 0;
  const newTotal = data.total ?? 0;

  if (prevVisible !== newVisible) {
    animateCount(visibleEl, newVisible);
    prevVisible = newVisible;
  } else {
    visibleEl.textContent = newVisible;
  }

  if (prevTotal !== newTotal) {
    animateCount(totalEl, newTotal);
    prevTotal = newTotal;
  } else {
    totalEl.textContent = newTotal;
  }

  document.querySelector('#processingFps').textContent = data.processing_fps ?? '0.0';

  const calState = document.querySelector('#calibrationState');
  const newState = data.background_calibrated ? 'ACTIVE' : 'NOT SET';
  calState.textContent = newState;
  calState.style.color = data.background_calibrated ? '#00ddb3' : '#f59e0b';
}

async function refreshStatus() {
  const response = await fetch('/api/status');
  const data = await response.json();
  showStatus(data.message, data.running);
  showMetrics(data);
}

/* ── Button Loading Helper ──────────────────────────────────────────────── */
function setLoading(button, loading) {
  button.classList.toggle('btn-loading', loading);
  button.disabled = loading;
}

/* ── Camera Controls ────────────────────────────────────────────────────── */
document.querySelector('#startButton').addEventListener('click', async () => {
  const btn = document.querySelector('#startButton');
  setLoading(btn, true);
  try {
    const response = await fetch('/api/camera/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
    });
    const data = await response.json();
    showStatus(data.message, data.ok);
    showToast(data.message, data.ok ? 'success' : 'error');
    if (data.ok) feed.src = `/api/frame.jpg?t=${Date.now()}`;
  } catch {
    showToast('Failed to reach the server.', 'error');
  } finally {
    setLoading(btn, false);
  }
});

document.querySelector('#stopButton').addEventListener('click', async () => {
  const btn = document.querySelector('#stopButton');
  setLoading(btn, true);
  try {
    const response = await fetch('/api/camera/stop', { method: 'POST' });
    const data = await response.json();
    showStatus(data.message, false);
    showToast(data.message, 'info');
  } catch {
    showToast('Failed to reach the server.', 'error');
  } finally {
    setLoading(btn, false);
  }
});

document.querySelector('#resetButton').addEventListener('click', async () => {
  const btn = document.querySelector('#resetButton');
  setLoading(btn, true);
  try {
    const response = await fetch('/api/count/reset', { method: 'POST' });
    const data = await response.json();
    showStatus(data.message, badge.classList.contains('active'));
    showToast('Ring count reset to zero.', 'success');
    prevTotal = 0;
    prevVisible = 0;
  } catch {
    showToast('Failed to reach the server.', 'error');
  } finally {
    setLoading(btn, false);
  }
});

document.querySelector('#calibrateButton').addEventListener('click', async () => {
  const btn = document.querySelector('#calibrateButton');
  setLoading(btn, true);
  try {
    const response = await fetch('/api/background/calibrate', { method: 'POST' });
    const data = await response.json();
    showStatus(data.message, badge.classList.contains('active'));
    showToast(data.message, data.ok ? 'success' : 'error');
  } catch {
    showToast('Failed to reach the server.', 'error');
  } finally {
    setLoading(btn, false);
  }
});

/* ── Field of View Sliders ──────────────────────────────────────────────── */
function updateSliderLabels() {
  sliderIds.forEach((id) => {
    const slider = document.querySelector(`#${id}`);
    document.querySelector(`#${id}Value`).textContent = `${slider.value}%`;
  });
}

sliderIds.forEach((id) => {
  document.querySelector(`#${id}`).addEventListener('input', updateSliderLabels);
});

async function loadSettings() {
  const response = await fetch('/api/settings');
  const data = await response.json();
  document.querySelector('#left').value = data.left;
  document.querySelector('#right').value = data.right;
  document.querySelector('#top').value = data.top;
  document.querySelector('#bottom').value = data.bottom;
  document.querySelector('#countingLine').value = data.counting_line;
  updateSliderLabels();
}

document.querySelector('#applyFovButton').addEventListener('click', async () => {
  const btn = document.querySelector('#applyFovButton');
  setLoading(btn, true);
  try {
    const response = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        left: document.querySelector('#left').value,
        right: document.querySelector('#right').value,
        top: document.querySelector('#top').value,
        bottom: document.querySelector('#bottom').value,
        counting_line: document.querySelector('#countingLine').value,
      }),
    });
    const data = await response.json();
    showStatus(data.message, badge.classList.contains('active'));
    showToast(data.ok ? 'Field of view updated.' : data.message, data.ok ? 'success' : 'error');
  } catch {
    showToast('Failed to reach the server.', 'error');
  } finally {
    setLoading(btn, false);
  }
});

/* ── Initialization ─────────────────────────────────────────────────────── */
refreshStatus().catch(() => showStatus('Could not reach the Python server.', false));
loadSettings().catch(() => {});
refreshFrame();
refreshDebugFrame();
updateClock();
setInterval(refreshFrame, 50);
setInterval(refreshDebugFrame, 700);
setInterval(() => refreshStatus().catch(() => showStatus('Could not reach the Python server.', false)), 1000);
setInterval(updateClock, 1000);
