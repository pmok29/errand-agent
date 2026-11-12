"use strict";

const $ = (id) => document.getElementById(id);
const NOISE = new Set(["context_sent", "assistant_msg"]);
const KIND_TAG = { template: "status", error: "status", notification: "update", card: "approval needed", memory_prompt: "memory" };
let busy = false;
let lastSeq = 0;   // the backlog and the live stream can overlap

// ---------- API ----------
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// ---------- chat ----------
function addMessage(m) {
  const div = document.createElement("div");
  const kind = m.role === "user" ? "user" : m.kind;
  div.className = `msg ${kind}`;
  if (KIND_TAG[kind]) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = KIND_TAG[kind];
    div.appendChild(tag);
  }
  div.appendChild(document.createTextNode(m.text));
  const cardCode = kind === "card" && m.card && m.card.code;
  if (cardCode) {
    retireCards();  // only the newest card can be confirmed from the UI; older codes are rejected server-side anyway
    const row = document.createElement("div");
    row.className = "card-actions";
    const btn = document.createElement("button");
    btn.className = "primary";
    btn.textContent = `Confirm`;
    btn.title = `Sends "CONFIRM ${cardCode}"`;
    btn.onclick = () => { btn.disabled = true; send(`CONFIRM ${cardCode}`); };
    const code = document.createElement("code");
    code.textContent = `CONFIRM ${cardCode}`;
    row.append(btn, code);
    div.appendChild(row);
  }
  if (kind === "memory_prompt") {
    const row = document.createElement("div");
    row.className = "card-actions";
    for (const answer of ["remember yes", "remember no"]) {
      const b = document.createElement("button");
      b.textContent = answer.replace("remember ", "").replace(/^./, (c) => c.toUpperCase());
      b.onclick = () => send(answer);
      row.appendChild(b);
    }
    div.appendChild(row);
  }
  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
}

function retireCards() {
  document.querySelectorAll(".msg.card button").forEach((b) => { b.disabled = true; });
}

function systemNote(text) {
  addMessage({ role: "assistant", kind: "system", text });
}

async function send(text) {
  if (busy || !text.trim()) return;
  busy = true;
  $("text").value = "";
  addMessage({ role: "user", kind: "chat", text });
  try {
    const r = await api("/api/message", { text });
    r.replies.forEach(addMessage);
    if (r.crashed) showBanner("The agent process crashed right after the provider answered the book call. Click “Restart agent” to recover.");
    setState(r.state);
  } catch (e) {
    systemNote(`Error: ${e.message}`);
  } finally {
    busy = false;
    $("text").focus();
  }
}

function showBanner(text) {
  $("banner").textContent = text;
  $("banner").hidden = !text;
}

// ---------- trace ----------
const short = (v, n = 90) => {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s && s.length > n ? s.slice(0, n) + "…" : s;
};
const META = new Set(["seq", "type", "ts", "run_id", "t_ms"]);
const fields = (e, skip = []) => Object.fromEntries(Object.entries(e).filter(([k]) => !META.has(k) && !skip.includes(k)));
const money = (c) => (c ? ` · ≈₹${c.inr.toFixed(2)}` : "");

function summarize(e) {
  switch (e.type) {
    case "run_start": return [`▶ ${e.kind}${e.job ? " " + e.job : ""}`, "info run"];
    case "run_end": {
      const t = e.totals || {};
      return [`■ ${e.outcome} · ${t.llm_calls} LLM calls · ${t.input_tokens}/${t.output_tokens} tok · ${t.tool_calls} tools · ${t.wall_ms} ms`, "info"];
    }
    case "user_msg": return [short(e.text, 140), ""];
    case "context_sent":
      return [`${e.est_tokens}/${e.budget} tok · ${e.memories_included.length} memories · history ${e.history_kept} kept/${e.history_dropped} dropped · ${e.redaction_hits} scrubbed`, ""];
    case "llm_call": {
      const calls = (e.tool_calls || []).map((c) => c.name).join(", ");
      return [`${e.model} · ${e.input_tokens}/${e.output_tokens} tok · ${Math.round(e.latency_ms)} ms${money(e.cost)} → ${calls || short(e.text, 60)}`, ""];
    }
    case "gate_decision":
      return [`${e.allow ? "✓" : "✗"} ${e.actor}: ${e.tool}${e.allow ? "" : " — " + e.reason}`, e.allow ? "" : "bad"];
    case "tool_call": {
      const status = e.timed_out ? "TIMEOUT" : e.ok === false || (e.result && e.result.ok === false)
        ? `error ${(e.error || e.result.error || {}).code}` : "ok";
      return [`${e.actor}: ${e.tool}(${short(e.args, 70)}) → ${status}`, status === "ok" ? "" : "warn"];
    }
    case "state_transition": return [`${e.frm} → ${e.to} (${e.reason})`, "ok"];
    case "request_updated": return [`changed ${e.changed.join(", ") || "nothing"} · ${short(e.request, 80)}`, ""];
    case "approval_created": return [`code ${e.code} · ${e.draft.action} ${e.draft.provider_name} ${e.draft.slot} ₹${e.draft.price}`, "info"];
    case "approval_consumed": return [`attempt ${e.attempt_no} · key ${e.idem_key.slice(0, 10)}…`, "ok"];
    case "approval_invalidated": case "approval_expired": return [e.reason || "expired", "warn"];
    case "execution_update": return [`${e.state} · attempt ${e.attempt_no} · retries ${e.retries}${e.last_error ? " · " + e.last_error : ""}`, ""];
    case "injection_flagged": return [`${e.tool}: withheld ${e.fields.join(", ")}`, "warn"];
    case "claim_blocked": return [`“${short(e.original, 60)}” → template`, "bad"];
    case "memory_write": case "memory_confirmed": return [`${e.key} = ${e.value}${e.status ? " (" + e.status + ")" : ""}`, "ok"];
    case "memory_rejected": return [`${e.key}${e.reason ? " — " + e.reason : ""}`, "warn"];
    case "memory_forgotten": return [`${e.key} (${e.count})${e.by ? " via " + e.by : ""}`, "warn"];
    case "limit_hit": return [`${e.limit} in ${e.state}`, "bad"];
    case "error": case "llm_error": return [short(e.error, 140), "bad"];
    case "llm_fallback": return [short(e.message, 140), "warn"];
    case "assistant_msg": return [`${e.kind}: ${short(e.text, 120)}`, ""];
    case "demo": return [`${e.action} ${short(fields(e, ["action"]), 100)}`, "info"];
    default: return [short(fields(e), 140), ""];
  }
}

function addEvent(e) {
  if (e.seq <= lastSeq) return;
  lastSeq = e.seq;
  const [sum, cls] = summarize(e);
  const row = document.createElement("div");
  row.className = `ev ${cls}`;
  row.dataset.noise = NOISE.has(e.type) ? "1" : "";
  if (row.dataset.noise && $("hideNoise").checked) row.hidden = true;
  const cells = [e.seq, e.t_ms != null ? `${Math.round(e.t_ms)}ms` : "", e.type, sum];
  ["seq", "t", "type", "sum"].forEach((c, i) => {
    const span = document.createElement("span");
    span.className = c;
    span.textContent = cells[i];
    row.appendChild(span);
  });
  row.onclick = () => {
    const open = row.querySelector("pre");
    if (open) return open.remove();
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(e, null, 2);
    row.appendChild(pre);
  };
  const log = $("trace");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  log.appendChild(row);
  if (atBottom) log.scrollTop = log.scrollHeight;
}

// ---------- state, memory ----------
function setState(s) {
  $("llm").textContent = s.llm;
  $("clock").textContent = s.now;
  $("errand").textContent = s.errand ? `${s.errand.kind} · ${s.errand.state}` : "no errand yet";
  $("crash").classList.toggle("armed", s.crash_next);
  $("crash").textContent = s.crash_next ? "Crash armed" : "Crash next booking";
  const sel = $("preset");
  if (!sel.options.length) s.presets.forEach((p) => sel.add(new Option(p, p)));
  sel.value = s.preset;
  $("seed").checked = s.seeded;
  if (!s.errand || s.errand.state !== "AWAITING_APPROVAL") retireCards();  // nothing to confirm
}

function renderMemories(list) {
  const groups = { usable: [], candidate: [], forgotten: [] };
  for (const m of list) {
    const g = m.status === "candidate" ? "candidate" : ["forgotten", "rejected"].includes(m.status) ? "forgotten" : "usable";
    groups[g].push(m);
  }
  const titles = { usable: "Used by the agent", candidate: "Inferred, waiting for your confirmation", forgotten: "Forgotten or declined" };
  const root = $("memories");
  root.replaceChildren();
  for (const [g, items] of Object.entries(groups)) {
    if (!items.length) continue;
    const box = document.createElement("div");
    box.className = "mem-group";
    const h = document.createElement("h4");
    h.textContent = `${titles[g]} (${items.length})`;
    box.appendChild(h);
    for (const m of items) {
      const row = document.createElement("div");
      row.className = "mem";
      const k = Object.assign(document.createElement("span"), { className: "k", textContent: m.key.replaceAll("_", " ") });
      const v = Object.assign(document.createElement("span"), { className: "v", textContent: m.value });
      const s = Object.assign(document.createElement("span"), { className: "s", textContent: `${m.source}${m.scope !== "global" ? " · " + m.scope : ""}` });
      row.append(k, v, s);
      if (g !== "forgotten") {
        const b = Object.assign(document.createElement("button"), { className: "small", textContent: "Forget" });
        b.onclick = async () => renderMemories((await api("/api/memories/forget", { key: m.key })).memories);
        row.appendChild(b);
      }
      box.appendChild(row);
    }
    root.appendChild(box);
  }
}

async function load() {
  const s = await api("/api/state");
  $("log").replaceChildren();
  $("trace").replaceChildren();
  lastSeq = 0;
  s.history.forEach(addMessage);
  s.trace.forEach(addEvent);
  setState(s);  // after the history, so its cards can be retired
  renderMemories(s.memories);
  if (!s.history.length) systemNote(`New session · scenario “${s.preset}”${s.seeded ? " · seeded profile (55 memories, past bookings)" : ""}`);
}

async function control(path, after) {
  try {
    const r = await api(path, {});
    if (r.state) setState(r.state);
    if (after) after(r);
  } catch (e) {
    systemNote(`Error: ${e.message}`);
  }
}

// ---------- wiring ----------
$("composer").onsubmit = (ev) => { ev.preventDefault(); send($("text").value); };
$("reset").onclick = async () => {
  showBanner("");
  await api("/api/demo/reset", { preset: $("preset").value, seed: $("seed").checked });
  await load();
};
$("cancel").onclick = () => control("/api/demo/provider-cancel", (r) => systemNote(`Demo: provider cancelled booking ${r.booking.booking_id}`));
$("advance").onclick = () => control("/api/demo/advance?seconds=600");
$("crash").onclick = () => control("/api/demo/crash-next");
$("restart").onclick = () => control("/api/demo/restart", () => { showBanner(""); systemNote("Agent restarted · recovery ran"); });
$("memToggle").onclick = async () => {
  $("drawer").hidden = !$("drawer").hidden;
  if (!$("drawer").hidden) renderMemories(await api("/api/memories"));
};
$("memClose").onclick = () => { $("drawer").hidden = true; };
$("hideNoise").onchange = () => {
  for (const row of $("trace").children) row.hidden = !!row.dataset.noise && $("hideNoise").checked;
};
$("clearTrace").onclick = () => $("trace").replaceChildren();  // lastSeq stays: cleared events don't come back

const stream = new EventSource("/api/stream");
stream.addEventListener("trace", (m) => addEvent(JSON.parse(m.data).event));
stream.addEventListener("chat", (m) => addMessage(JSON.parse(m.data).message));
setInterval(async () => { try { setState(await api("/api/status")); } catch {} }, 15000);  // clock + errand chip

load();
