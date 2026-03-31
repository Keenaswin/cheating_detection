/* ============================================================
   script.js — Live dashboard polling & UI updates
   ============================================================ */

"use strict";

const POLL_INTERVAL_MS = 1000;   // fetch /api/state every 1 s

// ── Element references ─────────────────────────────────────
const scoreValue  = document.getElementById("score-value");
const scoreBar    = document.getElementById("score-bar");
const riskLevel   = document.getElementById("risk-level");
const scoreCard   = document.getElementById("score-card");
const activeList  = document.getElementById("active-events");
const historyList = document.getElementById("history-list");
const histCount   = document.getElementById("history-count");
const connBadge   = document.getElementById("connection-badge");
const clock       = document.getElementById("clock");
const toastCont   = document.getElementById("toast-container");

// ── Track alerts already shown (avoid duplicate toasts) ────
let lastNewAlerts = [];

// ── Clock ──────────────────────────────────────────────────
function tickClock() {
  clock.textContent = new Date().toLocaleTimeString();
}
setInterval(tickClock, 1000);
tickClock();

// ── Colour helpers ─────────────────────────────────────────
function levelClass(level) {
  return { LOW: "low", MEDIUM: "medium", HIGH: "high" }[level] || "low";
}

// ── Update score card ──────────────────────────────────────
function updateScore(score, level) {
  const pct = Math.min(Math.max(score, 0), 100);
  const cls  = levelClass(level);

  scoreValue.textContent = Math.round(pct);
  scoreValue.className   = `score-value col-${cls}`;

  scoreBar.style.width   = pct + "%";
  scoreBar.className     = `score-bar-fill bar-${cls}`;

  riskLevel.textContent  = level;
  riskLevel.className    = `risk-level col-${cls}`;

  scoreCard.className    = `score-card level-${cls}`;
}

// ── Update active events list ──────────────────────────────
function updateEvents(events) {
  if (events.length === 0) {
    activeList.innerHTML = '<li class="event-item ok">✔ All clear</li>';
    return;
  }
  activeList.innerHTML = events
    .map(e => `<li class="event-item warning">⚠ ${e}</li>`)
    .join("");
}

// ── Update history list ────────────────────────────────────
function updateHistory(history) {
  histCount.textContent = history.length;

  if (history.length === 0) {
    historyList.innerHTML = '<p class="muted">No alerts yet.</p>';
    return;
  }

  // Show newest first
  const rows = [...history].reverse().map(item => {
    const cls    = levelClass(item.level);
    const labels = item.events.join(", ");
    return `
      <div class="history-item">
        <span class="hi-label col-${cls}">${labels}</span>
        <span class="hi-score col-${cls}">${Math.round(item.score)}</span>
        <span class="hi-time">${item.timestamp.slice(11)}</span>
      </div>`;
  });

  historyList.innerHTML = rows.join("");
}

// ── Toast notifications ────────────────────────────────────
function showToast(event) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `
    <div class="toast-title">⚠ Alert</div>
    <div>${event}</div>`;
  toastCont.appendChild(toast);
  // Remove after animation finishes (~4 s)
  setTimeout(() => toast.remove(), 4200);
}

// ── Connection badge ───────────────────────────────────────
function setConnected(ok) {
  if (ok) {
    connBadge.textContent = "● LIVE";
    connBadge.className   = "badge badge-ok";
  } else {
    connBadge.textContent = "● OFFLINE";
    connBadge.className   = "badge badge-err";
  }
}

// ── Main polling loop ──────────────────────────────────────
async function poll() {
  try {
    const res   = await fetch("/api/state");
    if (!res.ok) throw new Error("Non-200");
    const data  = await res.json();

    setConnected(true);
    updateScore(data.score, data.level);
    updateEvents(data.events);
    updateHistory(data.history);

    // Fire toasts only for genuinely new alerts
    if (data.new_alerts && data.new_alerts.length > 0) {
      const key = data.new_alerts.join("|") + data.timestamp;
      if (key !== lastNewAlerts) {
        data.new_alerts.forEach(showToast);
        lastNewAlerts = key;
      }
    }

  } catch (err) {
    setConnected(false);
    console.warn("[poll] Fetch failed:", err.message);
  }
}

// Start polling immediately, then every POLL_INTERVAL_MS
poll();
setInterval(poll, POLL_INTERVAL_MS);
