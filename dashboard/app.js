/* ───────────────────────────────────────────────────────────────────────────
   app.js — Dashboard logic for AI Agent Coordination & Decision Engine
   ─────────────────────────────────────────────────────────────────────────── */

const API_BASE = "http://localhost:8000";

/* ── Helpers ───────────────────────────────────────────────────────────────── */

function $(id) { return document.getElementById(id); }

function show(id) { $(id).style.display = ""; }
function hide(id) { $(id).style.display = "none"; }

/** Minimal markdown → HTML renderer (headings, bold, italic, code, lists, blockquote) */
function renderMarkdown(md) {
  if (!md) return "";
  let html = md
    // code blocks
    .replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) =>
      `<pre><code>${escapeHtml(code.trim())}</code></pre>`)
    // headings
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm,  "<h2>$1</h2>")
    .replace(/^# (.+)$/gm,   "<h1>$1</h1>")
    // blockquote
    .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
    // bold / italic
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g,     "<em>$1</em>")
    // inline code
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // unordered lists
    .replace(/^\s*[-*] (.+)$/gm, "<li>$1</li>")
    // ordered lists
    .replace(/^\s*\d+\. (.+)$/gm, "<li>$1</li>")
    // wrap consecutive <li> in <ul>
    .replace(/(<li>[\s\S]*?<\/li>)(\n<li>[\s\S]*?<\/li>)*/g, m => `<ul>${m}</ul>`)
    // paragraphs (double newlines)
    .replace(/\n{2,}/g, "</p><p>")
    // single line breaks
    .replace(/\n/g, "<br />");
  return `<p>${html}</p>`;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ── Agent colour helpers ─────────────────────────────────────────────────── */

const AGENT_COLORS = {
  ANALYST:   { bg: "rgba(139,92,246,0.2)",  color: "#8b5cf6", tag: "tag-analyst"   },
  EXECUTOR:  { bg: "rgba(245,158,11,0.2)",  color: "#f59e0b", tag: "tag-executor"  },
  RESPONDER: { bg: "rgba(34,197,94,0.2)",   color: "#22c55e", tag: "tag-responder" },
  DEFAULT:   { bg: "rgba(90,110,248,0.2)",  color: "#5a6ef8", tag: "tag-default"   },
};

function agentStyle(role) {
  return AGENT_COLORS[role?.toUpperCase()] || AGENT_COLORS.DEFAULT;
}

/* ── Loading states ──────────────────────────────────────────────────────── */

const LOADING_STEPS = [
  "Planner decomposing request…",
  "Analyst gathering insights…",
  "Executor performing tasks…",
  "Responder composing reply…",
];

let _loadingTimer = null;

function showLoading() {
  show("loading");
  $("run-btn").disabled = true;
  $("loading").classList.remove("hidden");
  let step = 0;
  $("loading-label").textContent = LOADING_STEPS[0];
  _loadingTimer = setInterval(() => {
    step = (step + 1) % LOADING_STEPS.length;
    $("loading-label").textContent = LOADING_STEPS[step];
  }, 2500);
}

function hideLoading() {
  clearInterval(_loadingTimer);
  $("loading").classList.add("hidden");
  $("run-btn").disabled = false;
}

/* ── Render Plan ─────────────────────────────────────────────────────────── */

function renderPlan(plan) {
  $("plan-goal").textContent = plan.goal || "—";
  const container = $("plan-tasks");
  container.innerHTML = "";

  const tasks = plan.tasks || [];
  tasks.forEach((task, i) => {
    const role = (task.agent || "DEFAULT").toUpperCase();
    const style = agentStyle(role);
    const item = document.createElement("div");
    item.className = "task-item";
    item.style.animationDelay = `${i * 80}ms`;
    item.innerHTML = `
      <div class="task-id" style="background:${style.bg};color:${style.color}">${task.id}</div>
      <div class="task-desc">${escapeHtml(task.description || "")}</div>
      <span class="task-agent-tag ${style.tag}">${role}</span>
    `;
    container.appendChild(item);
  });

  show("plan-card");
}

/* ── Render Task Results ─────────────────────────────────────────────────── */

function renderTaskResults(results) {
  const container = $("task-results");
  container.innerHTML = "";

  results.forEach((res, i) => {
    const agentName = res.agent || "Agent";
    const role      = agentName.replace("Agent", "").toUpperCase();
    const style     = agentStyle(role);
    const status    = res.status || res.confidence || "DONE";
    const statusColor = status === "SUCCESS" || status === "HIGH" ? "#22c55e" :
                        status === "FAILED"  || status === "LOW"  ? "#ef4444" : "#f59e0b";

    const item = document.createElement("div");
    item.className = "result-item";
    item.style.animationDelay = `${i * 100}ms`;

    // Summarise body (exclude agent key)
    const bodyData = Object.fromEntries(
      Object.entries(res).filter(([k]) => k !== "agent")
    );

    item.innerHTML = `
      <div class="result-header" onclick="toggleResult(this)">
        <span class="result-agent" style="color:${style.color}">● ${agentName}</span>
        <span class="result-status" style="color:${statusColor}">${status} ▾</span>
      </div>
      <div class="result-body">${JSON.stringify(bodyData, null, 2)}</div>
    `;
    container.appendChild(item);
  });

  show("results-card");
}

function toggleResult(header) {
  const body = header.nextElementSibling;
  body.classList.toggle("collapsed");
  const arrow = header.querySelector(".result-status");
  arrow.textContent = arrow.textContent.includes("▾")
    ? arrow.textContent.replace("▾", "▴")
    : arrow.textContent.replace("▴", "▾");
}

/* ── Render Final Response ────────────────────────────────────────────────── */

function renderResponse(text) {
  $("final-response").innerHTML = renderMarkdown(text);
  show("response-card");
}

/* ── Render History ──────────────────────────────────────────────────────── */

async function refreshHistory() {
  try {
    const res  = await fetch(`${API_BASE}/api/history`);
    const data = await res.json();
    const list = $("history-list");
    const msgs = data.history || [];

    if (!msgs.length) {
      list.innerHTML = '<p class="empty-msg">No messages yet. Run a query to get started.</p>';
      return;
    }

    list.innerHTML = msgs.map(m => {
      const agent   = m.agent ? ` <span style="opacity:.6">(${m.agent})</span>` : "";
      const preview = (m.content || "").slice(0, 180).replace(/\n/g, " ");
      return `
        <div class="history-item role-${m.role}">
          <div class="history-role">${m.role}${agent}</div>
          <div class="history-content">${escapeHtml(preview)}${m.content.length > 180 ? "…" : ""}</div>
        </div>`;
    }).join("");

    list.scrollTop = list.scrollHeight;
  } catch (_) { /* ignore if server not ready */ }
}

/* ── Clear History ────────────────────────────────────────────────────────── */

async function clearHistory() {
  try {
    await fetch(`${API_BASE}/api/history`, { method: "DELETE" });
    await refreshHistory();
  } catch (e) {
    alert("Could not reach the agent server. Make sure it's running.");
  }
}

/* ── Copy Response ────────────────────────────────────────────────────────── */

function copyResponse() {
  const text = $("final-response").innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event.target;
    btn.textContent = "✅ Copied!";
    setTimeout(() => (btn.textContent = "📋 Copy"), 1800);
  });
}

/* ── Set Query from Example ───────────────────────────────────────────────── */

function setQuery(text) {
  $("query-input").value = text;
  $("query-input").focus();
}

/* ── Main Run ─────────────────────────────────────────────────────────────── */

async function runQuery() {
  const query = $("query-input").value.trim();
  if (!query) {
    $("query-input").focus();
    return;
  }

  // Reset UI
  hide("plan-card");
  hide("results-card");
  hide("response-card");
  hide("error-card");

  showLoading();

  try {
    const res = await fetch(`${API_BASE}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        reset_memory: $("reset-memory").checked,
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Server error ${res.status}: ${err}`);
    }

    const data = await res.json();

    renderPlan(data.plan);
    renderTaskResults(data.task_results);
    renderResponse(data.response);

    $("session-badge").textContent = `Session: ${data.session_id}`;

    await refreshHistory();

  } catch (err) {
    $("error-body").textContent = err.message;
    show("error-card");
  } finally {
    hideLoading();
  }
}

/* ── Health Check ─────────────────────────────────────────────────────────── */

async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    const badge = $("health-badge");
    badge.textContent = "● Connected";
    badge.style.color = "#22c55e";
    $("session-badge").textContent = `Session: ${data.session_id}`;
  } catch (_) {
    const badge = $("health-badge");
    badge.textContent = "● Server Offline";
    badge.style.color = "#ef4444";
  }
}

/* ── Keyboard Shortcut (Ctrl/Cmd + Enter) ─────────────────────────────────── */

document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runQuery();
});

/* ── Init ─────────────────────────────────────────────────────────────────── */

checkHealth();
refreshHistory();
setInterval(checkHealth, 15000);
