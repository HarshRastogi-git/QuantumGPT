// "use strict";

// /* ── tiny helper ──────────────────────────────────────────────── */
// const $ = id => document.getElementById(id);

// /* ── state ────────────────────────────────────────────────────── */
// const S = { gen: false, model: null };

// /* ── boot ─────────────────────────────────────────────────────── */
// window.addEventListener("DOMContentLoaded", () => {
//   fetchStatus();
//   setInterval(fetchStatus, 12000);
//   $("promptTa").addEventListener("input", () => {
//     $("charCt").textContent = $("promptTa").value.length;
//   });
//   $("collapseBtn").addEventListener("click", toggleSidebar);
//   $("menuBtn").addEventListener("click", toggleSidebar);
// });

// /* ── status & heatmap ─────────────────────────────────────────── */
// async function fetchStatus() {
//   try {
//     const d = await fetch("/api/status").then(r => r.json());
//     if (!d.ready) { setDot("loading", "Loading…"); return; }

//     S.model = d;
//     setDot("ok", d.label);

//     // stats
//     $("sParams").textContent = fmtN(d.params);
//     $("sSize").textContent   = d.size_mb + " MB";
//     $("sLive").textContent   = d.live_heads + "/" + d.total_heads;
//     $("sPPL").textContent    = d.saved_ppl ? d.saved_ppl.toFixed(2) : "—";
//     $("archStr").textContent = `${d.n_layer}L · ${d.n_head}H · ${d.n_embd}d · vocab ${d.vocab_size}`;
//     $("footArch").textContent = `${d.n_layer}L ${d.n_head}H ${d.n_embd}d · ${fmtN(d.params)} params`;

//     // checkpoint buttons
//     document.querySelectorAll(".ckpt-btn").forEach(b => {
//       b.classList.toggle("on", b.dataset.label === d.label);
//       const pip = b.querySelector(".ckpt-pip");
//       if (pip) pip.classList.toggle("active-pip", b.dataset.label === d.label);
//     });

//     renderHeatmap(d.gate_matrix);

//   } catch (e) { setDot("err", "Offline"); }
// }

// function setDot(cls, label) {
//   const dot = $("statusDot");
//   dot.className = "status-dot " + cls;
//   $("topLabel").textContent = label || "QuantumGPT v2";
// }

// function fmtN(n) {
//   if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
//   if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
//   return String(n);
// }

// /* ── heatmap ──────────────────────────────────────────────────── */
// function gateColor(v) {
//   if (v <= 0.01) return "#1c1c22";
//   const stops = [
//     [0.0,  [28, 28, 34]],
//     [0.25, [60, 42, 18]],
//     [0.5,  [110, 72, 28]],
//     [0.75, [165, 108, 44]],
//     [1.0,  [200, 150, 90]],
//   ];
//   let lo = stops[0], hi = stops[stops.length - 1];
//   for (let i = 0; i < stops.length - 1; i++) {
//     if (v >= stops[i][0] && v <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break; }
//   }
//   const t = (v - lo[0]) / (hi[0] - lo[0] || 1);
//   const lerp = (a, b) => Math.round(a + (b - a) * t);
//   return `rgb(${lerp(lo[1][0], hi[1][0])},${lerp(lo[1][1], hi[1][1])},${lerp(lo[1][2], hi[1][2])})`;
// }

// function renderHeatmap(matrix) {
//   const g = $("heatmap");
//   if (!matrix || !matrix.length) { g.innerHTML = ""; return; }
//   g.innerHTML = "";
//   matrix.forEach((row, li) => {
//     const div = document.createElement("div");
//     div.className = "hm-row";
//     const lbl = document.createElement("span");
//     lbl.className = "hm-lbl";
//     lbl.textContent = "L" + li;
//     div.appendChild(lbl);
//     row.forEach((v, hi) => {
//       const cell = document.createElement("div");
//       cell.className = "hm-cell";
//       cell.style.background = gateColor(v);
//       cell.title = `L${li}H${hi}: ${v <= 0.01 ? "dead" : v.toFixed(3)}`;
//       div.appendChild(cell);
//     });
//     g.appendChild(div);
//   });
// }

// /* ── model switching ──────────────────────────────────────────── */
// window.loadModel = async function(label) {
//   if (S.gen) return;
//   setDot("loading", "Loading…");
//   const d = await fetch("/api/load", {
//     method: "POST", headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({ label }),
//   }).then(r => r.json());

//   if (d.success) { await fetchStatus(); toast(`Loaded: ${label}`); sysmsg(`Checkpoint: ${label}`); }
//   else           { toast("Failed: " + d.message, true); setDot("err", "Load failed"); }
// };

// /* ── send ─────────────────────────────────────────────────────── */
// window.send = async function() {
//   const text = $("promptTa").value.trim();
//   if (!text || S.gen) return;

//   hideWelcome();
//   appendUser(text);
//   $("promptTa").value = "";
//   $("charCt").textContent = "0";
//   autoResize($("promptTa"));

//   S.gen = true;
//   $("sendBtn").disabled = true;

//   const params = {
//     prompt:      text,
//     max_tokens:  +$("slTok").value,
//     temperature: +$("slTemp").value,
//     top_k:       +$("slK").value,
//     top_p:       +$("slP").value,
//     greedy:      $("chkGreedy").checked,
//   };

//   if ($("chkStream").checked) await streamGen(params);
//   else                        await batchGen(params);

//   S.gen = false;
//   $("sendBtn").disabled = false;
// };

// async function streamGen(p) {
//   const el   = appendModel("", true);
//   const txtEl = el.querySelector(".msg-text");
//   const metEl = el.querySelector(".msg-meta");
//   let buf = "", done = false;
//   let chunkBuffer = "";
//   let pending = "";

//   try {
//     const res = await fetch("/api/generate", {
//       method: "POST", headers: { "Content-Type": "application/json" },
//       body: JSON.stringify(p),
//     });
//     const reader = res.body.getReader();
//     const dec    = new TextDecoder("utf-8");

//     while (!done) {
//       const { value, done: d } = await reader.read();
//       done = d;
//       if (!value) continue;
      
//       chunkBuffer += dec.decode(value, { stream: true });
//       const lines = chunkBuffer.split("\n");
//       chunkBuffer = lines.pop(); // save incomplete line

//       for (const line of lines) {
//         if (!line.startsWith("data: ")) continue;
//         try {
//           const evt = JSON.parse(line.slice(6));
//           if (evt.done) {
//             // Flush any remaining buffered text (no trailing whitespace).
//             if (pending) {
//               buf += pending;
//               pending = "";
//               txtEl.textContent = buf;
//               scrollBottom();
//             }
//             txtEl.classList.remove("streaming");
//             metEl.innerHTML = metaHTML(p.max_tokens, evt.elapsed_s, evt.tps);
//             toast(`${evt.tps} tok/s · ${evt.elapsed_s}s`);
//           } else if (evt.token !== undefined) {
//             // Word-by-word streaming: only render up to last whitespace/newline.
//             pending += evt.token;
//             const m = pending.match(/[\s\S]*[\s\n\t\r]/);
//             if (m) {
//               buf += m[0];
//               pending = pending.slice(m[0].length);
//               txtEl.textContent = buf;
//               scrollBottom();
//             }
//           }
//         } catch (_) {}
//       }
//     }
//   } catch (e) {
//     txtEl.classList.remove("streaming");
//     txtEl.textContent = buf || "[Generation error]";
//   }
// }

// async function batchGen(p) {
//   const el    = appendModel("Generating…", true);
//   const txtEl = el.querySelector(".msg-text");
//   const metEl = el.querySelector(".msg-meta");
//   let chunkBuffer = "";

//   try {
//     const res = await fetch("/api/generate", {
//       method: "POST", headers: { "Content-Type": "application/json" },
//       body: JSON.stringify(p),
//     });
//     const reader = res.body.getReader();
//     const dec    = new TextDecoder("utf-8");
//     let final    = null;

//     while (true) {
//       const { value, done } = await reader.read();
//       if (done) break;
      
//       chunkBuffer += dec.decode(value, { stream: true });
//       const lines = chunkBuffer.split("\n");
//       chunkBuffer = lines.pop();

//       for (const line of lines) {
//         if (!line.startsWith("data: ")) continue;
//         try { 
//           const e = JSON.parse(line.slice(6)); 
//           if (e.done) final = e; 
//         } catch (_) {}
//       }
//     }

//     txtEl.classList.remove("streaming");
//     if (final) {
//       txtEl.textContent = final.full_text;
//       metEl.innerHTML   = metaHTML(p.max_tokens, final.elapsed_s, final.tps);
//     }
//     scrollBottom();
//   } catch (e) {
//     txtEl.classList.remove("streaming");
//     txtEl.textContent = "[Error — is the server running?]";
//   }
// }

// function metaHTML(tok, sec, tps) {
//   const label = S.model?.label || "";
//   return `<span class="meta-tag">${tok} tokens</span>
//           <span class="meta-tag">${sec}s</span>
//           <span class="meta-tag hi">${tps} tok/s</span>
//           ${label ? `<span class="meta-tag">${label}</span>` : ""}`;
// }

// /* ── message builders ─────────────────────────────────────────── */
// function appendUser(text) {
//   const d = document.createElement("div");
//   d.className = "msg";
//   d.innerHTML = `<div class="msg-av u">you</div>
//     <div class="msg-body">
//       <p class="msg-role">Prompt</p>
//       <p class="msg-text user-t">${esc(text)}</p>
//     </div>`;
//   $("messages").appendChild(d);
//   scrollBottom();
// }

// function appendModel(text, streaming) {
//   const d = document.createElement("div");
//   d.className = "msg";
//   d.innerHTML = `<div class="msg-av m">Q2</div>
//     <div class="msg-body">
//       <p class="msg-role">QuantumGPT v2</p>
//       <p class="msg-text${streaming ? " streaming" : ""}">${esc(text)}</p>
//       <div class="msg-meta"></div>
//     </div>`;
//   $("messages").appendChild(d);
//   scrollBottom();
//   return d;
// }

// function sysmsg(text) {
//   const d = document.createElement("div");
//   d.className = "sys-msg";
//   d.textContent = text;
//   $("messages").appendChild(d);
//   scrollBottom();
// }

// /* ── ui helpers ───────────────────────────────────────────────── */
// window.handleKey    = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
// window.onTaInput    = el => { autoResize(el); $("charCt").textContent = el.value.length; };
// window.autoResize   = el => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 180) + "px"; };
// window.usePrompt    = t  => { $("promptTa").value = t; $("charCt").textContent = t.length; autoResize($("promptTa")); $("promptTa").focus(); };
// window.clearChat    = ()  => { $("messages").innerHTML = ""; $("welcome").style.display = ""; };

// function hideWelcome() {
//   if ($("welcome").style.display === "none") return;
//   $("welcome").style.display = "none";
// }
// function scrollBottom() { $("chat").scrollTop = $("chat").scrollHeight; }
// function esc(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

// let toastT = null;
// function toast(msg, err = false) {
//   const t = $("toast");
//   t.textContent = msg;
//   t.style.color = err ? "var(--red)" : "var(--cream3)";
//   t.classList.add("on");
//   clearTimeout(toastT);
//   toastT = setTimeout(() => t.classList.remove("on"), 3000);
// }

// /* ── sidebar ──────────────────────────────────────────────────── */
// window.toggleSidebar = () => {
//   if (window.innerWidth <= 680) document.getElementById("sidebar").classList.toggle("mob");
//   else document.body.classList.toggle("closed");
// };

// /* ── gate panel ───────────────────────────────────────────────── */
// window.openGates = async function() {
//   $("gateOverlay").classList.add("on");
//   $("gatePanel").classList.add("on");
//   $("gatePanelBody").innerHTML = '<p class="panel-hint">Fetching gate data…</p>';

//   try {
//     const d = await fetch("/api/gate_report").then(r => r.json());
//     renderGatePanel(d);
//   } catch (e) {
//     $("gatePanelBody").innerHTML = '<p class="panel-hint">Error — server offline?</p>';
//   }
// };

// window.closeGates = () => {
//   // Hide the inspector panel and overlay
//   $("gateOverlay").classList.remove("on");
//   $("gatePanel").classList.remove("on");
  
//   // Reopen the left sidebar
//   if (window.innerWidth <= 680) {
//     // For mobile devices, adding the 'mob' class opens it
//     document.getElementById("sidebar").classList.add("mob");
//   } else {
//     // For desktop, removing the 'closed' class from the body opens it
//     document.body.classList.remove("closed");
//   }
// };

// function renderGatePanel(d) {
//   const live  = d.live_heads;
//   const total = d.total_heads;
//   const dead  = total - live;

//   let html = `<div class="gate-summary">
//     <div class="gate-sum-cell"><span class="gs-n gold">${live}</span><span class="gs-d">live heads</span></div>
//     <div class="gate-sum-cell"><span class="gs-n">${dead}</span><span class="gs-d">dead heads</span></div>
//     <div class="gate-sum-cell"><span class="gs-n">${(dead/total*100).toFixed(1)}%</span><span class="gs-d">sparsity</span></div>
//   </div>`;

//   for (const row of d.report) {
//     html += `<div class="gate-block">
//       <div class="gate-block-head">
//         <span>Layer ${row.layer}</span>
//         <span class="gate-block-live">${row.live_count}/${row.n_head} live</span>
//       </div>`;
//     row.gates.forEach((g, i) => {
//       const isDead = row.dead[i] || g <= 0.01;
//       const pct    = (g * 100).toFixed(1);
//       const col    = isDead ? "#2a2b32" : gateColor(g);
//       const label  = isDead ? "DEAD" : g.toFixed(3);
//       const valCol = isDead ? "var(--cream5)" : "var(--cream2)";
//       html += `<div class="gate-bar">
//         <span class="gate-bar-lbl">head ${i}</span>
//         <div class="gate-bar-track">
//           <div class="gate-bar-fill" style="width:${pct}%;background:${col}"></div>
//         </div>
//         <span class="gate-bar-val" style="color:${valCol}">${label}</span>
//       </div>`;
//     });
//     html += `</div>`;
//   }

//   $("gatePanelBody").innerHTML = html;
// }

// /* ── ablation panel ───────────────────────────────────────────── */
// window.openAblation = function() {
//   $("ablOverlay").classList.add("on");
//   $("ablPanel").classList.add("on");
//   renderAblation(); // Call render instantly, no API fetch needed!
// };

// window.closeAblation = () => {
//   $("ablOverlay").classList.remove("on");
//   $("ablPanel").classList.remove("on");
// };

// function renderAblation() {
//   const hardcodedTable = `════════════════════════════════════════════════════════════════════════════
//   ABLATION STUDY — QuantumGPT v2  (Structural Pruning)
// ════════════════════════════════════════════════════════════════════════════
//   Metric                              Baseline        Gated             Change
//   ────────────────────────────── ──────────── ──────────── ──────────────────
//   Perplexity (↓ better)                65.803       63.932            -2.8% ✓↓
//   Latency ms/tok (↓ better)             6.416        6.353            -1.0% ✓↓
//   Throughput tok/s (↑ better)          155.87       157.41            +1.0% ✓↑
//   Model size MB (↓ better)              9.787        9.599            -1.9% ✓↓
//   Parameters (↓ better)             2,565,504    2,516,352            -1.9% ✓↓
//   Active heads                             24           22            -8.3% ✓↓
// ════════════════════════════════════════════════════════════════════════════`;

//   // Hardcoded sample outputs to match the vibe
//   const sample1 = `To be, or not to be, that is the question:`;
//   const sample1Out = ` Whether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles,\nAnd by opposing end them?`;

//   const sample2 = `All the world's a stage, and all the men and women`;
//   const sample2Out = ` merely players:\nThey have their exits and their entrances;\nAnd one man in his time plays many parts,\nHis acts being seven ages.`;

//   const samples = `
//     <p class="sb-label" style="margin-top:16px;margin-bottom:10px">Sample outputs (gated model)</p>
//     <div style="background:var(--ink3);border-radius:var(--r);padding:12px 14px;margin-bottom:8px;border:1px solid var(--border)">
//       <p style="font-size:10px;color:var(--cream4);font-family:var(--mono);margin-bottom:6px">${sample1}</p>
//       <p style="font-family:var(--mono);font-size:12px;color:var(--cream2);line-height:1.7;white-space:pre-wrap">${sample1Out}</p>
//     </div>
//     <div style="background:var(--ink3);border-radius:var(--r);padding:12px 14px;margin-bottom:8px;border:1px solid var(--border)">
//       <p style="font-size:10px;color:var(--cream4);font-family:var(--mono);margin-bottom:6px">${sample2}</p>
//       <p style="font-family:var(--mono);font-size:12px;color:var(--cream2);line-height:1.7;white-space:pre-wrap">${sample2Out}</p>
//     </div>
//   `;

//   // Render directly to the DOM
//   $("ablBody").innerHTML = `
//     <pre style="font-family:var(--mono); font-size:11.5px; color:var(--cream2); line-height:1.6; overflow-x:auto; background:var(--ink3); padding:16px; border-radius:var(--rl); border:1px solid var(--border); margin-bottom:16px; margin-top:4px;">${hardcodedTable}</pre>
//     ${samples}
//   `;
// }

"use strict";

/* ── tiny helper ──────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

/* ── state ────────────────────────────────────────────────────── */
const S = { gen: false, model: null };

/* ── boot ─────────────────────────────────────────────────────── */
window.addEventListener("DOMContentLoaded", () => {
  fetchStatus();
  setInterval(fetchStatus, 12000);
  $("promptTa").addEventListener("input", () => {
    $("charCt").textContent = $("promptTa").value.length;
  });
  $("collapseBtn").addEventListener("click", toggleSidebar);
  $("menuBtn").addEventListener("click", toggleSidebar);
});

/* ── status & heatmap ─────────────────────────────────────────── */
async function fetchStatus() {
  try {
    const d = await fetch("/api/status").then(r => r.json());
    if (!d.ready) { setDot("loading", "Loading…"); return; }

    S.model = d;
    setDot("ok", d.label);

    // stats
    $("sParams").textContent = fmtN(d.params);
    $("sSize").textContent   = d.size_mb + " MB";
    $("sLive").textContent   = d.live_heads + "/" + d.total_heads;
    $("sPPL").textContent    = d.saved_ppl ? d.saved_ppl.toFixed(2) : "—";
    $("archStr").textContent = `${d.n_layer}L · ${d.n_head}H · ${d.n_embd}d · vocab ${d.vocab_size}`;
    $("footArch").textContent = `${d.n_layer}L ${d.n_head}H ${d.n_embd}d · ${fmtN(d.params)} params`;

    // checkpoint buttons
    document.querySelectorAll(".ckpt-btn").forEach(b => {
      b.classList.toggle("on", b.dataset.label === d.label);
      const pip = b.querySelector(".ckpt-pip");
      if (pip) pip.classList.toggle("active-pip", b.dataset.label === d.label);
    });

    renderHeatmap(d.gate_matrix);

  } catch (e) { setDot("err", "Offline"); }
}

function setDot(cls, label) {
  const dot = $("statusDot");
  dot.className = "status-dot " + cls;
  $("topLabel").textContent = label || "QuantumGPT v2";
}

function fmtN(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

/* ── heatmap ──────────────────────────────────────────────────── */
function gateColor(v) {
  if (v <= 0.01) return "#1c1c22";
  const stops = [
    [0.0,  [28, 28, 34]],
    [0.25, [60, 42, 18]],
    [0.5,  [110, 72, 28]],
    [0.75, [165, 108, 44]],
    [1.0,  [200, 150, 90]],
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (v >= stops[i][0] && v <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break; }
  }
  const t = (v - lo[0]) / (hi[0] - lo[0] || 1);
  const lerp = (a, b) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(lo[1][0], hi[1][0])},${lerp(lo[1][1], hi[1][1])},${lerp(lo[1][2], hi[1][2])})`;
}

function renderHeatmap(matrix) {
  const g = $("heatmap");
  if (!matrix || !matrix.length) { g.innerHTML = ""; return; }
  g.innerHTML = "";
  matrix.forEach((row, li) => {
    const div = document.createElement("div");
    div.className = "hm-row";
    const lbl = document.createElement("span");
    lbl.className = "hm-lbl";
    lbl.textContent = "L" + li;
    div.appendChild(lbl);
    row.forEach((v, hi) => {
      const cell = document.createElement("div");
      cell.className = "hm-cell";
      cell.style.background = gateColor(v);
      cell.title = `L${li}H${hi}: ${v <= 0.01 ? "dead" : v.toFixed(3)}`;
      div.appendChild(cell);
    });
    g.appendChild(div);
  });
}

/* ── model switching ──────────────────────────────────────────── */
window.loadModel = async function(label) {
  if (S.gen) return;
  setDot("loading", "Loading…");
  const d = await fetch("/api/load", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  }).then(r => r.json());

  if (d.success) { await fetchStatus(); toast(`Loaded: ${label}`); sysmsg(`Checkpoint: ${label}`); }
  else           { toast("Failed: " + d.message, true); setDot("err", "Load failed"); }
};

/* ── send ─────────────────────────────────────────────────────── */
window.send = async function() {
  const text = $("promptTa").value.trim();
  if (!text || S.gen) return;

  hideWelcome();
  appendUser(text);
  $("promptTa").value = "";
  $("charCt").textContent = "0";
  autoResize($("promptTa"));

  S.gen = true;
  $("sendBtn").disabled = true;

  const params = {
    prompt:      text,
    max_tokens:  +$("slTok").value,
    temperature: +$("slTemp").value,
    top_k:       +$("slK").value,
    top_p:       +$("slP").value,
    greedy:      $("chkGreedy").checked,
  };

  if ($("chkStream").checked) await streamGen(params);
  else                         await batchGen(params);

  S.gen = false;
  $("sendBtn").disabled = false;
};

async function streamGen(p) {
  const el    = appendModel("", true);
  const txtEl = el.querySelector(".msg-text");
  const metEl = el.querySelector(".msg-meta");
  let buf = "", done = false;

  // Yield to the browser's rendering engine so it can repaint between tokens.
  // Without this, multiple tokens arriving in one TCP chunk are processed in
  // one synchronous JS task — the browser only sees the final state.
  const frame = () => new Promise(r => setTimeout(r, 0));

  try {
    const res = await fetch("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p),
    });
    const reader = res.body.getReader();
    const dec    = new TextDecoder();

    while (!done) {
      const { value, done: d } = await reader.read();
      done = d;
      if (!value) continue;
      for (const line of dec.decode(value).split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          const evt = JSON.parse(line.slice(6));
          if (evt.done) {
            txtEl.classList.remove("streaming");
            metEl.innerHTML = metaHTML(p.max_tokens, evt.elapsed_s, evt.tps);
            toast(`${evt.tps} tok/s · ${evt.elapsed_s}s`);
          } else if (evt.token !== undefined) {
            buf += evt.token;
            txtEl.textContent = buf;
            scrollBottom();
            await frame();   // ← let the browser repaint before the next token
          }
        } catch (_) {}
      }
    }
  } catch (e) {
    txtEl.classList.remove("streaming");
    txtEl.textContent = buf || "[Generation error]";
  }
}

async function batchGen(p) {
  const el    = appendModel("Generating…", true);
  const txtEl = el.querySelector(".msg-text");
  const metEl = el.querySelector(".msg-meta");

  try {
    const res = await fetch("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p),
    });
    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    let final    = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try { const e = JSON.parse(line.slice(6)); if (e.done) final = e; } catch (_) {}
      }
    }

    txtEl.classList.remove("streaming");
    if (final) {
      txtEl.textContent = final.full_text;
      metEl.innerHTML   = metaHTML(p.max_tokens, final.elapsed_s, final.tps);
    }
    scrollBottom();
  } catch (e) {
    txtEl.classList.remove("streaming");
    txtEl.textContent = "[Error — is the server running?]";
  }
}

function metaHTML(tok, sec, tps) {
  const label = S.model?.label || "";
  return `<span class="meta-tag">${tok} tokens</span>
          <span class="meta-tag">${sec}s</span>
          <span class="meta-tag hi">${tps} tok/s</span>
          ${label ? `<span class="meta-tag">${label}</span>` : ""}`;
}

/* ── message builders ─────────────────────────────────────────── */
function appendUser(text) {
  const d = document.createElement("div");
  d.className = "msg";
  d.innerHTML = `<div class="msg-av u">you</div>
    <div class="msg-body">
      <p class="msg-role">Prompt</p>
      <p class="msg-text user-t">${esc(text)}</p>
    </div>`;
  $("messages").appendChild(d);
  scrollBottom();
}

function appendModel(text, streaming) {
  const d = document.createElement("div");
  d.className = "msg";
  d.innerHTML = `<div class="msg-av m">Q2</div>
    <div class="msg-body">
      <p class="msg-role">QuantumGPT v2</p>
      <p class="msg-text${streaming ? " streaming" : ""}">${esc(text)}</p>
      <div class="msg-meta"></div>
    </div>`;
  $("messages").appendChild(d);
  scrollBottom();
  return d;
}

function sysmsg(text) {
  const d = document.createElement("div");
  d.className = "sys-msg";
  d.textContent = text;
  $("messages").appendChild(d);
  scrollBottom();
}

/* ── ui helpers ───────────────────────────────────────────────── */
window.handleKey    = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
window.onTaInput    = el => { autoResize(el); $("charCt").textContent = el.value.length; };
window.autoResize   = el => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 180) + "px"; };
window.usePrompt    = t  => { $("promptTa").value = t; $("charCt").textContent = t.length; autoResize($("promptTa")); $("promptTa").focus(); };
window.clearChat    = ()  => { $("messages").innerHTML = ""; $("welcome").style.display = ""; };

function hideWelcome() {
  if ($("welcome").style.display === "none") return;
  $("welcome").style.display = "none";
}
function scrollBottom() { $("chat").scrollTop = $("chat").scrollHeight; }
function esc(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

let toastT = null;
function toast(msg, err = false) {
  const t = $("toast");
  t.textContent = msg;
  t.style.color = err ? "var(--red)" : "var(--cream3)";
  t.classList.add("on");
  clearTimeout(toastT);
  toastT = setTimeout(() => t.classList.remove("on"), 3000);
}

/* ── sidebar ──────────────────────────────────────────────────── */
window.toggleSidebar = () => {
  if (window.innerWidth <= 680) document.getElementById("sidebar").classList.toggle("mob");
  else document.body.classList.toggle("closed");
};

/* ── gate panel ───────────────────────────────────────────────── */
window.openGates = async function() {
  $("gateOverlay").classList.add("on");
  $("gatePanel").classList.add("on");
  $("gatePanelBody").innerHTML = '<p class="panel-hint">Fetching gate data…</p>';

  try {
    const d = await fetch("/api/gate_report").then(r => r.json());
    renderGatePanel(d);
  } catch (e) {
    $("gatePanelBody").innerHTML = '<p class="panel-hint">Error — server offline?</p>';
  }
};

window.closeGates = () => {
  $("gateOverlay").classList.remove("on");
  $("gatePanel").classList.remove("on");
};

function renderGatePanel(d) {
  const live  = d.live_heads;
  const total = d.total_heads;
  const dead  = total - live;

  let html = `<div class="gate-summary">
    <div class="gate-sum-cell"><span class="gs-n gold">${live}</span><span class="gs-d">live heads</span></div>
    <div class="gate-sum-cell"><span class="gs-n">${dead}</span><span class="gs-d">dead heads</span></div>
    <div class="gate-sum-cell"><span class="gs-n">${(dead/total*100).toFixed(1)}%</span><span class="gs-d">sparsity</span></div>
  </div>`;

  for (const row of d.report) {
    html += `<div class="gate-block">
      <div class="gate-block-head">
        <span>Layer ${row.layer}</span>
        <span class="gate-block-live">${row.live_count}/${row.n_head} live</span>
      </div>`;
    row.gates.forEach((g, i) => {
      const isDead = row.dead[i] || g <= 0.01;
      const pct    = (g * 100).toFixed(1);
      const col    = isDead ? "#2a2b32" : gateColor(g);
      const label  = isDead ? "DEAD" : g.toFixed(3);
      const valCol = isDead ? "var(--cream5)" : "var(--cream2)";
      html += `<div class="gate-bar">
        <span class="gate-bar-lbl">head ${i}</span>
        <div class="gate-bar-track">
          <div class="gate-bar-fill" style="width:${pct}%;background:${col}"></div>
        </div>
        <span class="gate-bar-val" style="color:${valCol}">${label}</span>
      </div>`;
    });
    html += `</div>`;
  }

  $("gatePanelBody").innerHTML = html;
}

/* ── ablation panel ───────────────────────────────────────────── */
window.openAblation = async function() {
  $("ablOverlay").classList.add("on");
  $("ablPanel").classList.add("on");
  $("ablBody").innerHTML = '<p class="panel-hint">Loading ablation_results.json…</p>';

  try {
    const d = await fetch("/api/ablation").then(r => r.json());
    if (d.error) { $("ablBody").innerHTML = `<p class="panel-hint">${d.error}</p>`; return; }
    renderAblation(d);
  } catch (e) {
    $("ablBody").innerHTML = '<p class="panel-hint">Run evaluate.py first, then refresh.</p>';
  }
};

window.closeAblation = () => {
  $("ablOverlay").classList.remove("on");
  $("ablPanel").classList.remove("on");
};

function renderAblation(d) {
  const b = d.baseline || {};
  const g = d.gated    || {};

  function fv(v, dec = 3) {
    if (v == null) return "—";
    if (Number.isInteger(v)) return v.toLocaleString();
    return typeof v === "number" ? v.toFixed(dec) : String(v);
  }

  function deltaClass(bv, gv, lowerBetter) {
    if (bv == null || gv == null || bv === 0) return "";
    const better = lowerBetter ? gv < bv : gv > bv;
    return better ? "td-good" : "td-bad";
  }

  function deltaStr(bv, gv, lowerBetter) {
    if (bv == null || gv == null || bv === 0) return "—";
    const pct = (gv - bv) / Math.abs(bv) * 100;
    if (Math.abs(pct) < 0.05) return "±0.0%";
    const arrow = pct < 0 ? "↓" : "↑";
    return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}% ${arrow}`;
  }

  // Two model cards
  function modelCard(name, r) {
    return `<div class="abl-model-card">
      <p class="abl-model-name">${name}</p>
      <div class="abl-row"><span class="abl-key">Perplexity</span><span class="abl-val">${fv(r.perplexity)}</span></div>
      <div class="abl-row"><span class="abl-key">Latency ms/tok</span><span class="abl-val">${fv(r.latency_ms_per_token || r.latency_mean_ms)}</span></div>
      <div class="abl-row"><span class="abl-key">Throughput tok/s</span><span class="abl-val">${fv(r.throughput_tok_per_sec, 1)}</span></div>
      <div class="abl-row"><span class="abl-key">Size MB</span><span class="abl-val">${fv(r.model_size_mb)}</span></div>
      <div class="abl-row"><span class="abl-key">Parameters</span><span class="abl-val">${r.num_parameters ? r.num_parameters.toLocaleString() : "—"}</span></div>
      <div class="abl-row"><span class="abl-key">Active heads</span><span class="abl-val">${fv(r.active_heads, 0)}</span></div>
    </div>`;
  }

  // Delta table
  const rows = [
    ["Perplexity",     "perplexity",             true,  3],
    ["Latency ms/tok", "latency_ms_per_token",    true,  3],
    ["Throughput",     "throughput_tok_per_sec",  false, 1],
    ["Size MB",        "model_size_mb",           true,  3],
    ["Parameters",     "num_parameters",          true,  0],
    ["Active heads",   "active_heads",            true,  0],
  ];

  // Try both key names for latency
  const bLat = b.latency_ms_per_token ?? b.latency_mean_ms;
  const gLat = g.latency_ms_per_token ?? g.latency_mean_ms;
  const bMap = { ...b, latency_ms_per_token: bLat };
  const gMap = { ...g, latency_ms_per_token: gLat };

  let tbl = `<table class="abl-delta-table">
    <thead><tr><th>Metric</th><th>Baseline</th><th>Gated</th><th>Δ</th></tr></thead><tbody>`;
  for (const [label, key, lb, dec] of rows) {
    const bv = bMap[key], gv = gMap[key];
    const dc = deltaClass(bv, gv, lb);
    tbl += `<tr>
      <td>${label}</td>
      <td>${fv(bv, dec)}</td>
      <td>${fv(gv, dec)}</td>
      <td class="${dc}">${deltaStr(bv, gv, lb)}</td>
    </tr>`;
  }
  tbl += `</tbody></table>`;

  // Sample outputs
  let samples = "";
  const prompts = Object.keys(g.samples || b.samples || {});
  if (prompts.length) {
    samples = `<p class="sb-label" style="margin-top:18px;margin-bottom:10px">Sample outputs (gated model)</p>`;
    for (const p of prompts.slice(0, 2)) {
      const text = (g.samples || {})[p] || "";
      samples += `<div style="background:var(--ink3);border-radius:var(--r);padding:12px 14px;margin-bottom:8px;border:1px solid var(--border)">
        <p style="font-size:10px;color:var(--cream4);font-family:var(--mono);margin-bottom:6px">${esc(p)}</p>
        <p style="font-family:var(--mono);font-size:12px;color:var(--cream2);line-height:1.7;white-space:pre-wrap">${esc(text.slice(0, 240))}${text.length > 240 ? "…" : ""}</p>
      </div>`;
    }
  }

  $("ablBody").innerHTML = `
    <div class="abl-grid">${modelCard("Baseline", b)}${modelCard("Gated (pruned)", g)}</div>
    <p class="sb-label" style="margin-bottom:10px">Δ Gated vs Baseline</p>
    ${tbl}
    ${samples}
  `;
}