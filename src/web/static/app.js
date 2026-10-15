/* Newsletter Agent front end: start a run, poll it, approve or request changes. */

const POLL_MS = 1000;

const els = {
  form: document.getElementById("run-form"),
  goal: document.getElementById("goal"),
  runButton: document.getElementById("run-button"),
  status: document.getElementById("status"),
  health: document.getElementById("health"),
  log: document.getElementById("log"),
  reviewPanel: document.getElementById("review-panel"),
  critique: document.getElementById("critique"),
  reviewItems: document.getElementById("review-items"),
  feedback: document.getElementById("feedback"),
  approve: document.getElementById("approve-button"),
  revise: document.getElementById("revise-button"),
  resultPanel: document.getElementById("result-panel"),
  resultMeta: document.getElementById("result-meta"),
  emailBody: document.getElementById("email-body"),
  htmlPreview: document.getElementById("html-preview"),
  placeholder: document.getElementById("placeholder"),
};

const state = { threadId: null, timer: null, status: "idle", logLines: 0 };

/* -- health ---------------------------------------------------------------- */

async function loadHealth() {
  try {
    const health = await getJson("/api/health");
    els.health.textContent = `${health.model} @ ${hostOf(health.provider)}`;
    if (!health.has_api_key) {
      els.health.textContent += " · no key";
      els.health.className = "health warn";
    } else {
      els.health.className = "health ok";
    }
  } catch {
    els.health.textContent = "backend unreachable";
    els.health.className = "health warn";
  }
}

/* -- run lifecycle --------------------------------------------------------- */

async function startRun(event) {
  event.preventDefault();
  const goal = els.goal.value.trim();
  if (goal.length < 3) return;

  stopPolling();
  resetPanes();
  setStatus("running", "starting…");

  try {
    const payload = await postJson("/api/runs", { goal, mode: selectedMode() });
    state.threadId = payload.thread_id;
    render(payload);
    startPolling();
  } catch (error) {
    setStatus("error", String(error.message || error));
    els.runButton.disabled = false;
  }
}

function startPolling() {
  state.timer = setInterval(poll, POLL_MS);
}

function stopPolling() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

async function poll() {
  if (!state.threadId) return;
  try {
    const payload = await getJson(`/api/runs/${state.threadId}`);
    render(payload);
    if (payload.status === "done" || payload.status === "error") stopPolling();
  } catch (error) {
    setStatus("error", String(error.message || error));
    stopPolling();
  }
}

async function sendDecision(action) {
  if (!state.threadId) return;
  els.approve.disabled = true;
  els.revise.disabled = true;
  try {
    const payload = await postJson(`/api/runs/${state.threadId}/resume`, {
      action,
      feedback: els.feedback.value,
    });
    els.feedback.value = "";
    render(payload);
    startPolling();
  } catch (error) {
    setStatus("error", String(error.message || error));
  }
}

/* -- rendering ------------------------------------------------------------- */

function render(payload) {
  setStatus(payload.status, describe(payload));
  appendLog(payload.log || []);

  const paused = payload.status === "paused" && payload.review;
  els.reviewPanel.classList.toggle("hidden", !paused);
  els.placeholder.classList.toggle("hidden", paused || Boolean(payload.result));

  if (paused) renderReview(payload.review);
  if (payload.result) renderResult(payload.result);

  const busy = payload.status === "running";
  els.runButton.disabled = busy;
  if (!paused) {
    els.approve.disabled = true;
    els.revise.disabled = true;
  } else {
    els.approve.disabled = false;
    els.revise.disabled = !payload.review.can_request_changes;
    els.revise.title = payload.review.can_request_changes
      ? ""
      : "Revision budget exhausted; approve to finish.";
  }
}

function renderReview(review) {
  const { critique } = review;
  els.critique.textContent =
    `self-critique  score ${critique.score}/10  verdict ${critique.verdict}\n` +
    `revisions used  ${review.revision_count} (${review.remaining_revisions} left)\n` +
    `items           ${review.item_count}\n` +
    (critique.issues.length ? `issues\n${critique.issues.map((i) => `  - ${i}`).join("\n")}` : "issues          none");

  els.reviewItems.replaceChildren(...review.draft.items.map(reviewItem));
}

function reviewItem(item) {
  const wrapper = document.createElement("article");
  wrapper.className = "item";

  const title = document.createElement("h3");
  title.textContent = item.title;
  wrapper.append(title);

  const link = document.createElement("a");
  link.href = item.url;
  link.textContent = item.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  wrapper.append(link);

  const summary = document.createElement("p");
  summary.textContent = item.summary;
  wrapper.append(summary);

  if (item.why_it_matters) {
    const why = document.createElement("p");
    why.className = "why";
    why.textContent = `Why it matters: ${item.why_it_matters}`;
    wrapper.append(why);
  }
  return wrapper;
}

function renderResult(result) {
  els.resultPanel.classList.remove("hidden");
  els.resultMeta.textContent =
    `subject    ${result.subject}\n` +
    `items      ${result.items.length}\n` +
    `revisions  ${result.revisions}\n` +
    `html       ${result.html_path}\n` +
    `email      ${result.email_path}`;
  els.emailBody.textContent = result.email_body || result.markdown;

  // Sandboxed: the newsletter HTML is agent-generated and gets no privileges.
  els.htmlPreview.setAttribute("sandbox", "");
  els.htmlPreview.srcdoc = result.html;
}

function appendLog(lines) {
  if (lines.length < state.logLines) state.logLines = 0;
  for (const line of lines.slice(state.logLines)) {
    const entry = document.createElement("li");
    entry.textContent = line;
    els.log.append(entry);
  }
  state.logLines = lines.length;
  els.log.scrollTop = els.log.scrollHeight;
}

function resetPanes() {
  els.log.replaceChildren();
  els.reviewItems.replaceChildren();
  els.resultMeta.textContent = "";
  els.emailBody.textContent = "";
  els.htmlPreview.removeAttribute("srcdoc");
  els.reviewPanel.classList.add("hidden");
  els.resultPanel.classList.add("hidden");
  els.placeholder.classList.remove("hidden");
  state.logLines = 0;
  state.status = "running";
}

function setStatus(status, text) {
  state.status = status;
  els.status.textContent = `${status} — ${text}`;
  els.status.className = `status ${status}`;
}

function describe(payload) {
  if (payload.error) return payload.error;
  if (payload.status === "running") return "working…";
  if (payload.status === "paused") return `waiting for your review at ${payload.next_node}`;
  if (payload.status === "done") return `${payload.result?.items.length ?? 0} items ready`;
  return payload.status;
}

/* -- helpers --------------------------------------------------------------- */

function selectedMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "autonomous";
}

function hostOf(url) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

/* -- wiring ---------------------------------------------------------------- */

els.form.addEventListener("submit", startRun);
els.approve.addEventListener("click", () => sendDecision("approve"));
els.revise.addEventListener("click", () => sendDecision("revise"));

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tab")) other.classList.toggle("active", other === tab);
    const showHtml = tab.dataset.tab === "html";
    els.htmlPreview.classList.toggle("hidden", !showHtml);
    els.emailBody.classList.toggle("hidden", showHtml);
  });
}

loadHealth();
