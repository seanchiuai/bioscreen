// ═══════════════════════════════════════════════════════════════════════════
// BioScreen — Vanilla JS Frontend
// ═══════════════════════════════════════════════════════════════════════════

// ── Section A: Constants & State ────────────────────────────────────────

const API_BASE = '/api';

let sessionId = crypto.randomUUID();
let currentResult = null;
let functionPollInterval = null;

const DEMO_SEQUENCES = {
  "Scorpion toxin Aah4 (HIGH — known toxin, 84aa)": {
    seq: "MNYLIMFSLALLLVIGVESGRDGYIVDSKNCTYFCGRNAYCNEECTKLKGESGYCQWASPYGNACYCYKLPDHVRTKGPGRCH",
    id: "Aah4-scorpion"
  },
  "Irditoxin snake venom (HIGH — in DB, 109aa)": {
    seq: "MKTLLLAVAVVAFVCLGSADQLGLGRQQIDWGQGQAVGPPYTLCFECNRMTSSDCSTALRCYRGSCYTLYRPDENCELKWAVKGCAETCPTAGPNERVKCCRSPRCNDD",
    id: "Irditoxin-A0S864"
  },
  "Spider toxin Beta-diguetoxin (HIGH — structural match, 74aa)": {
    seq: "ACVNDDYRSYYCVRKYMECGAEKSVGCWEYKAYQSCYCRQFAYKGEEGRPCVCRDFDGGQALKLHAGKEDSFH",
    id: "Beta-diguetoxin"
  },
  "AI-redesigned snake venom (MEDIUM — 39% identity, BLAST misses)": {
    seq: "APGRWRCEVWWSAGRCGNQPDAQMYPEKKKQCESPPLSECHKQWNRFDTEYECTSGCWY",
    id: "AI-redesigned-venom"
  },
  "AI-redesigned irditoxin (MEDIUM — 23% identity, BLAST misses)": {
    seq: "AAEAAAAEAAAAAAAAAAEAGTAAAPAPPPAAPAAPAPPPITYCYVCNRSLSSDCSTCQPCINGVCYIRYEKNANGEMVPVERGCSATCPTPGPNEKVICCTSDCCNSE",
    id: "AI-redesigned-irditoxin"
  },
  "Human lysozyme (LOW — benign enzyme, 130aa)": {
    seq: "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
    id: "Human-lysozyme"
  },
  "GFP (LOW — jellyfish fluorescent protein, 238aa)": {
    seq: "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    id: "GFP"
  },
  "Human insulin B chain (LOW — essential hormone, 30aa)": {
    seq: "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    id: "Insulin-B-chain"
  }
};

const DEMO_CONVERGENCE_SERIES = [
  ["Convergence query 1 (benign start)", "KKFERCQDTRTLKRLAMQGMPNISLANWMCLAKWISGYNTRANKYNGGRRSTDYGICQIWSRYWCNIGKTPGAVSCCGLSCSALLEDNMQAAVTCALRVNRDPQMINAWVWWSNRCSNTDVVQYVQGCGD"],
  ["Convergence query 2 (benign)", "MSICCEISFGVVSIEVELDEMVHRGIFVVSGEGSGDAAYRVLTHKFICWTGKTGDPEPTLDTTFSMPLQCFYRYHDNMKQVDFFKVEMEEGLVQKRCQFMKDRGNDKTCAEFNFEGDILVNRIEL"],
  ["Convergence query 3 (benign)", "GVLSMADKTGVCARWGKHGDHTGEYGKKYLERMFLSQVTTKQMFPLFDLSHPSAQVKLWGKKVADALTYAVAHHCSMPNALSALADLHATKLRFDPVNFKLLSHCLLVILTARLMAEITHQVHQSLDWFNPS"],
  ["Convergence query 4 (transitioning)", "QVIKAPRYPAQRKHPQDVNDCIAIEHKGEGELPHILYWNNMQERQTCVRWIDRRCMMHKYCAHQICSAIIWVQENCKHVGSWKLCLGVCFSASMCSDDCLVYYTRQYEH"],
  ["Convergence query 5 (transitioning)", "KGVAYCLRIVWHVFCLGYWDRPGSYQYDWAWKGRQWSGQEYSPCRKCTRVCNSCLSYGADCEFLKSVHLFRQTALCSLSVPMKGQSPFTPEATPFFKPMCCMSPDLNQD"],
  ["Convergence query 6 (toxin-like)", "MKGLLCNIWVGPIWKAGLARQLQLLPQPWLPWQGQAVGPPNHFCFEWFLQFSSTCSSAMRIYRGSCPVNYRNIYNCEDKLACKSVRVFCPTKNRNENNKRYRMFRCWIF"],
  ["Convergence query 7 (toxin-like)", "MKFLLLANAMKQFVCLTYTASLELGDFKIDWYQYEAVGPRYREKFEDERFTSSECSCALWCSEVKCYTLYRPKKNAFLFWATQGCAECCDQAKTQWRVPCHRSEQINDD"],
  ["Convergence query 8 (near-toxin)", "CKTMLLAVAVYAFVQLGSADQTGLNRQQEHFGQSQAVGPPYTLCFECNRRTLDDCSTALRCYRGCCYTGRRPDENCEMEWAYKGCNETCDTAGPNERVKCCRHPRCLDV"],
];

const COLOR_RED = "#dc3545";
const COLOR_ORANGE = "#d48c0e";
const COLOR_GREEN = "#2a9d5c";
const COLOR_TEXT = "#1a1a1a";
const COLOR_MUTED = "#6b6560";

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "DM Sans, sans-serif", color: COLOR_TEXT },
  margin: { l: 40, r: 40, t: 40, b: 40 },
};

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };


// ── Section B: API Client ────────────────────────────────────────────

async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(10000) });
    if (resp.ok) return await resp.json();
    return { status: "error", message: `API returned ${resp.status}` };
  } catch (e) {
    return { status: "error", message: `Cannot connect: ${e.message}` };
  }
}

async function screenSequence(sequence, sequenceId, topK, sid) {
  const payload = { sequence, sequence_id: sequenceId || null, top_k: topK };
  const resp = await fetch(`${API_BASE}/screen`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-Id": sid },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(180000),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  return await resp.json();
}

async function pollFunction(sequenceId) {
  try {
    const resp = await fetch(`${API_BASE}/function/${sequenceId}`, { signal: AbortSignal.timeout(10000) });
    if (resp.status === 200) return await resp.json();
    return null;
  } catch { return null; }
}

async function getSessionAlerts(sid) {
  try {
    const resp = await fetch(`${API_BASE}/session/${sid}/alerts`, { signal: AbortSignal.timeout(10000) });
    if (resp.ok) return await resp.json();
    return null;
  } catch { return null; }
}

async function compareStructures(queryPdb, targetUniprotId) {
  const resp = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_pdb: queryPdb, target_uniprot_id: targetUniprotId }),
    signal: AbortSignal.timeout(60000),
  });
  if (!resp.ok) return null;
  return await resp.json();
}


// ── Section C: View Switching ────────────────────────────────────────

function showView(viewId) {
  const views = ['single-screen', 'session-analysis'];
  views.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (v === viewId) {
      el.classList.remove('view-hidden');
    } else {
      el.classList.add('view-hidden');
    }
  });
  // Update nav buttons
  document.getElementById('nav-single').classList.toggle('active', viewId === 'single-screen');
  document.getElementById('nav-session').classList.toggle('active', viewId === 'session-analysis');
  // Update hash
  location.hash = viewId;
}


// ── Section D: Single Screen Logic ──────────────────────────────────

function initSingleScreen() {
  const select = document.getElementById('demo-select');
  for (const [label, data] of Object.entries(DEMO_SEQUENCES)) {
    const opt = document.createElement('option');
    opt.value = label;
    opt.textContent = label;
    select.appendChild(opt);
  }

  select.addEventListener('change', () => {
    const entry = DEMO_SEQUENCES[select.value];
    if (entry) {
      document.getElementById('seq-input').value = entry.seq;
      document.getElementById('seq-id-input').value = entry.id;
    }
  });
}

async function handleScreen() {
  const seq = document.getElementById('seq-input').value.trim();
  const seqId = document.getElementById('seq-id-input').value.trim();
  const topK = parseInt(document.getElementById('topk-input').value) || 5;

  if (!seq) { alert('Please enter a protein sequence.'); return; }
  if (seq.length < 10) { alert('Sequence too short (minimum 10 residues).'); return; }

  const btn = document.getElementById('btn-screen');
  btn.disabled = true;
  showLoading('Screening sequence... This may take 10-120 seconds for structure prediction.');

  try {
    const data = await screenSequence(seq, seqId, topK, sessionId);
    currentResult = data;
    document.getElementById('results-section').classList.remove('view-hidden');
    renderAllTabs(data);
    startFunctionPolling(data.sequence_id, data);
    showTab('overview', document.querySelector('[data-tab="overview"]'));
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
    btn.disabled = false;
  }
}

function showLoading(text) {
  let overlay = document.getElementById('loading-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `<div class="loading-spinner"></div><div class="loading-text">${escapeHtml(text)}</div>`;
    document.body.appendChild(overlay);
  } else {
    overlay.querySelector('.loading-text').textContent = text;
    overlay.style.display = 'flex';
  }
}

function hideLoading() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.style.display = 'none';
}

function showError(msg) {
  const section = document.getElementById('results-section');
  section.classList.remove('view-hidden');
  section.innerHTML = `<div class="error-box"><strong>Screening Error:</strong> ${escapeHtml(msg)}</div>`;
}

function startFunctionPolling(sequenceId, data) {
  if (functionPollInterval) clearInterval(functionPollInterval);
  if (!sequenceId) return;

  const btnFunc = document.getElementById('tab-btn-function');

  functionPollInterval = setInterval(async () => {
    const result = await pollFunction(sequenceId);
    if (result) {
      clearInterval(functionPollInterval);
      functionPollInterval = null;
      data.function_prediction = result;
      currentResult = data;
      // Show function tab button
      if (btnFunc) btnFunc.classList.remove('view-hidden');
      renderFunctionTab(data);
    }
  }, 5000);
}


// ── Section E: Tab Switching ────────────────────────────────────────

function showTab(tabId, btn) {
  // Hide all tab contents
  document.querySelectorAll('#results-section .tab-content').forEach(el => {
    el.classList.add('view-hidden');
  });
  // Show selected tab
  const tab = document.getElementById(`tab-${tabId}`);
  if (tab) tab.classList.remove('view-hidden');

  // Update tab bar buttons
  document.querySelectorAll('#results-section .tab-bar .tab-item').forEach(el => {
    el.classList.remove('active');
  });
  if (btn) btn.classList.add('active');

  // Re-render Plotly charts when tab becomes visible (Plotly needs visible container)
  if (tabId === 'overview' && currentResult) {
    setTimeout(() => {
      Plotly.Plots.resize(document.getElementById('chart-radar'));
      Plotly.Plots.resize(document.getElementById('chart-donut'));
      Plotly.Plots.resize(document.getElementById('chart-matches-bar'));
    }, 50);
  }
  if (tabId === 'matches' && currentResult) {
    setTimeout(() => {
      const el = document.getElementById('chart-heatmap');
      if (el && el.data) Plotly.Plots.resize(el);
    }, 50);
  }
  if (tabId === 'scores' && currentResult) {
    setTimeout(() => {
      const w = document.getElementById('chart-waterfall');
      const t = document.getElementById('chart-threshold');
      if (w && w.data) Plotly.Plots.resize(w);
      if (t && t.data) Plotly.Plots.resize(t);
    }, 50);
  }
}

function showMiniTab(prefix, tabId, btn) {
  document.querySelectorAll(`[data-mini-group="${prefix}"]`).forEach(el => {
    el.classList.add('view-hidden');
  });
  const tab = document.getElementById(`${prefix}-${tabId}`);
  if (tab) tab.classList.remove('view-hidden');

  // Update buttons
  const bar = btn.closest('.mini-tab-bar');
  if (bar) {
    bar.querySelectorAll('.mini-tab-item').forEach(b => b.classList.remove('active'));
  }
  btn.classList.add('active');
}


// ── Section F: Render All Tabs ──────────────────────────────────────

function renderAllTabs(data) {
  renderOverviewTab(data);
  renderMatchesTab(data);
  renderStructureTab(data);
  renderScoreBreakdownTab(data);
  renderExplainTab(data);
  if (data.function_prediction) {
    document.getElementById('tab-btn-function').classList.remove('view-hidden');
    renderFunctionTab(data);
  }
}


// ── Section G: Overview Tab ─────────────────────────────────────────

function renderOverviewTab(data) {
  const container = document.getElementById('tab-overview');
  const riskScore = data.risk_score;
  const riskLevel = data.risk_level;
  const factors = data.risk_factors || {};
  const topMatch = (data.top_matches || [])[0] || {};
  const matchName = escapeHtml(topMatch.name || "No match");
  const matchOrg = escapeHtml(topMatch.organism || "");
  const embSim = topMatch.embedding_similarity || 0;
  const strSim = topMatch.structure_similarity;
  const bestSim = Math.max(embSim, strSim || 0);
  const simLabel = (strSim && strSim >= embSim) ? "structure" : "embedding";

  const riskClass = riskLevel === 'HIGH' ? 'high' : riskLevel === 'MEDIUM' ? 'medium' : 'low';
  const pct = Math.round(riskScore * 100);

  container.innerHTML = `
    <div class="cards-row animate-in">
      <div class="summary-card card-risk">
        <div class="card-label">Risk Score</div>
        <div class="card-value risk-${riskClass}">${riskScore.toFixed(3)}</div>
        <div class="risk-bar"><div class="risk-bar-fill ${riskClass}" style="width:${pct}%"></div></div>
        <span class="risk-badge ${riskClass}">${riskLevel}</span>
      </div>
      <div class="summary-card card-match">
        <div class="card-label">Top Match</div>
        <div class="match-name">${matchName}</div>
        <div class="match-organism">${matchOrg}</div>
        <div class="match-score">${bestSim.toFixed(3)}<span class="match-score-type">${simLabel}</span></div>
      </div>
      <div class="summary-card card-mode">
        <div class="card-label">Screening Mode</div>
        <div class="mode-value">Full Pipeline</div>
        <div class="mode-badges">
          <span class="mode-badge">Embedding</span>
          <span class="mode-badge">Structure</span>
          <span class="mode-badge">Function</span>
        </div>
      </div>
    </div>

    <div class="chart-row">
      <div class="chart-container">
        <div class="chart-title">Threat Profile</div>
        <div id="chart-radar" style="width:100%;height:300px;"></div>
      </div>
      <div class="chart-container">
        <div class="chart-title">Score Contributions</div>
        <div id="chart-donut" style="width:100%;height:300px;"></div>
      </div>
    </div>

    <div class="chart-container">
      <div class="chart-title">Top Matches — Embedding vs Structure Similarity</div>
      <div id="chart-matches-bar" style="width:100%;height:300px;"></div>
    </div>
  `;

  setTimeout(() => {
    renderRadarChart('chart-radar', factors);
    renderDonutChart('chart-donut', factors, riskScore);
    renderMatchesBarChart('chart-matches-bar', data.top_matches || []);
  }, 50);
}


// ── Section H: Chart Renderers ──────────────────────────────────────

function renderRadarChart(containerId, factors) {
  const categories = ["Embedding Sim", "Structure Sim", "Function Overlap", "Active Site", "Session Anomaly"];
  const values = [
    factors.max_embedding_similarity || 0,
    factors.max_structure_similarity || 0,
    factors.function_overlap || 0,
    factors.active_site_overlap || 0,
    factors.session_anomaly_score || 0,
  ];

  const trace = {
    type: 'scatterpolar',
    r: [...values, values[0]],
    theta: [...categories, categories[0]],
    fill: 'toself',
    fillcolor: 'rgba(220,53,69,0.15)',
    line: { color: COLOR_RED, width: 2 },
    marker: { size: 5, color: COLOR_RED },
  };

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    polar: { radialaxis: { visible: true, range: [0, 1], tickfont: { size: 10 } } },
    showlegend: false,
    margin: { l: 60, r: 60, t: 30, b: 30 },
  };

  Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

function renderDonutChart(containerId, factors, riskScore) {
  const emb = factors.max_embedding_similarity || 0;
  const struct = factors.max_structure_similarity;
  const func = factors.function_overlap || 0;

  let labels, rawVals, weights, colors;
  if (struct != null) {
    weights = [0.50, 0.30, 0.20];
    labels = ["Embedding", "Structure", "Function"];
    rawVals = [emb, struct, func];
    colors = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN];
  } else {
    weights = [0.65, 0.35];
    labels = ["Embedding", "Function"];
    rawVals = [emb, func];
    colors = [COLOR_RED, COLOR_GREEN];
  }

  let contributions = rawVals.map((r, i) => r * weights[i]);
  if (contributions.every(c => c === 0)) contributions = contributions.map(() => 0.001);

  const trace = {
    type: 'pie',
    labels,
    values: contributions,
    hole: 0.55,
    marker: { colors },
    textinfo: 'label+percent',
    textfont: { size: 11 },
    hovertemplate: '%{label}: %{value:.3f}<extra></extra>',
  };

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    showlegend: false,
    margin: { l: 20, r: 20, t: 20, b: 20 },
    annotations: [{
      text: `<b>${riskScore.toFixed(2)}</b>`,
      x: 0.5, y: 0.5,
      font: { size: 22, color: COLOR_TEXT },
      showarrow: false,
    }],
  };

  Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

function renderMatchesBarChart(containerId, topMatches) {
  if (!topMatches || !topMatches.length) return;

  const names = topMatches.map(m => (m.name || '?').substring(0, 20));
  const embSims = topMatches.map(m => m.embedding_similarity || 0);
  const structSims = topMatches.map(m => m.structure_similarity || 0);

  const traces = [
    { name: 'Embedding Sim', x: names, y: embSims, type: 'bar', marker: { color: COLOR_TEXT } },
    { name: 'Structure Sim', x: names, y: structSims, type: 'bar', marker: { color: COLOR_RED } },
  ];

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    barmode: 'group',
    yaxis: { range: [0, 1], title: 'Similarity' },
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 },
    margin: { l: 50, r: 20, t: 30, b: 60 },
  };

  Plotly.newPlot(containerId, traces, layout, PLOTLY_CONFIG);
}

function renderSimilarityHeatmap(containerId, topMatches) {
  if (!topMatches || !topMatches.length) return;

  const names = topMatches.map(m => (m.name || '?').substring(0, 20));
  const embRow = topMatches.map(m => m.embedding_similarity || 0);
  const structRow = topMatches.map(m => m.structure_similarity || 0);

  const z = [embRow, structRow];
  const yLabels = ["Embedding Sim", "Structure Sim"];

  const annotations = [];
  z.forEach((row, i) => {
    row.forEach((val, j) => {
      annotations.push({
        x: names[j], y: yLabels[i],
        text: val.toFixed(2),
        font: { color: val > 0.5 ? 'white' : COLOR_TEXT, size: 11 },
        showarrow: false,
      });
    });
  });

  const trace = {
    type: 'heatmap',
    z, x: names, y: yLabels,
    colorscale: [[0, '#fee'], [0.5, '#f88'], [1, COLOR_RED]],
    zmin: 0, zmax: 1,
    showscale: true,
    colorbar: { title: 'Sim', len: 0.8 },
  };

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    annotations,
    yaxis: { autorange: 'reversed' },
    margin: { l: 100, r: 20, t: 30, b: 60 },
  };

  Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

function renderWaterfallChart(containerId, factors, riskScore) {
  const emb = factors.max_embedding_similarity || 0;
  const struct = factors.max_structure_similarity;
  const func = factors.function_overlap || 0;

  let items;
  if (struct != null) {
    items = [
      ["Embedding", emb * 0.50],
      ["Structure", struct * 0.30],
      ["Function", func * 0.20],
    ];
  } else {
    items = [
      ["Embedding", emb * 0.65],
      ["Function", func * 0.35],
    ];
  }

  const totalWeighted = items.reduce((s, [_, v]) => s + v, 0);
  const bonus = Math.max(0, riskScore - Math.min(1.0, totalWeighted));

  const labels = items.map(i => i[0]);
  const values = items.map(i => i[1]);
  const measures = items.map(() => "relative");
  const colors = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN].slice(0, items.length);

  if (bonus > 0.01) {
    labels.push("Synergy");
    values.push(bonus);
    measures.push("relative");
    colors.push("#a78bfa");
  }

  labels.push("Total");
  values.push(riskScore);
  measures.push("total");
  colors.push(COLOR_TEXT);

  const trace = {
    type: 'waterfall',
    x: labels, y: values,
    measure: measures,
    connector: { line: { color: COLOR_MUTED, width: 1 } },
    textposition: 'outside',
    text: values.map(v => v.toFixed(3)),
    marker: { color: colors },
  };

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    yaxis: { range: [0, Math.max(riskScore * 1.3, 0.3)], title: 'Score' },
    showlegend: false,
    margin: { l: 50, r: 20, t: 30, b: 40 },
  };

  Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

function renderThresholdChart(containerId, riskScore) {
  const traces = [
    { x: [0.3], y: ['Risk'], orientation: 'h', type: 'bar', marker: { color: COLOR_GREEN }, showlegend: false, hoverinfo: 'skip' },
    { x: [0.25], y: ['Risk'], orientation: 'h', type: 'bar', marker: { color: COLOR_ORANGE }, showlegend: false, hoverinfo: 'skip' },
    { x: [0.45], y: ['Risk'], orientation: 'h', type: 'bar', marker: { color: COLOR_RED }, showlegend: false, hoverinfo: 'skip' },
    {
      x: [riskScore], y: ['Risk'], mode: 'markers+text', type: 'scatter',
      marker: { size: 16, color: COLOR_TEXT, symbol: 'diamond' },
      text: [riskScore.toFixed(2)], textposition: 'top center',
      textfont: { size: 12, color: COLOR_TEXT },
      showlegend: false,
      hovertemplate: `Risk Score: ${riskScore.toFixed(3)}<extra></extra>`,
    },
  ];

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    barmode: 'stack',
    xaxis: { range: [0, 1], showticklabels: false },
    yaxis: { showticklabels: false },
    height: 100,
    margin: { l: 10, r: 10, t: 30, b: 30 },
    annotations: [
      { x: 0.15, y: -0.4, text: 'Low', showarrow: false, font: { size: 10, color: COLOR_GREEN }, yref: 'paper' },
      { x: 0.425, y: -0.4, text: 'Medium', showarrow: false, font: { size: 10, color: COLOR_ORANGE }, yref: 'paper' },
      { x: 0.775, y: -0.4, text: 'High', showarrow: false, font: { size: 10, color: COLOR_RED }, yref: 'paper' },
    ],
  };

  Plotly.newPlot(containerId, traces, layout, PLOTLY_CONFIG);
}

function renderFunctionBars(containerId, functionPrediction) {
  if (!functionPrediction) return;

  const items = [];
  for (const term of (functionPrediction.go_terms || [])) {
    let label = term.term || '?';
    if (term.name) label += ` — ${term.name}`;
    items.push([label, parseFloat(term.confidence || 0)]);
  }
  for (const ec of (functionPrediction.ec_numbers || [])) {
    items.push([`EC ${ec.number || '?'}`, parseFloat(ec.confidence || 0)]);
  }
  if (!items.length) return;

  items.sort((a, b) => b[1] - a[1]);

  const trace = {
    type: 'bar',
    y: items.map(i => i[0]),
    x: items.map(i => i[1]),
    orientation: 'h',
    marker: { color: COLOR_RED },
    text: items.map(i => i[1].toFixed(2)),
    textposition: 'auto',
  };

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    xaxis: { range: [0, 1], title: 'Confidence' },
    yaxis: { autorange: 'reversed' },
    height: Math.max(200, 40 * items.length + 80),
    margin: { l: 200, r: 20, t: 20, b: 40 },
  };

  Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}


// ── Section I: Matches Tab ──────────────────────────────────────────

function renderMatchesTab(data) {
  const container = document.getElementById('tab-matches');
  const matches = data.top_matches || [];
  const hasStructure = data.pdb_string != null;

  if (!matches.length) {
    container.innerHTML = '<div class="info-box">No significant matches found.</div>';
    return;
  }

  let tableRows = '';
  matches.forEach((m, i) => {
    const embPct = Math.round((m.embedding_similarity || 0) * 100);
    const strPct = Math.round((m.structure_similarity || 0) * 100);
    tableRows += `
      <tr>
        <td class="rank">${i + 1}</td>
        <td><strong>${escapeHtml(m.name || '')}</strong></td>
        <td>${escapeHtml(m.organism || '')}</td>
        <td>${escapeHtml(m.toxin_type || '')}</td>
        <td>
          <div class="sim-bar-cell">
            <div class="sim-bar"><div class="sim-bar-fill" style="width:${embPct}%"></div></div>
            <span class="sim-num">${(m.embedding_similarity || 0).toFixed(3)}</span>
          </div>
        </td>
        <td>
          <div class="sim-bar-cell">
            <div class="sim-bar"><div class="sim-bar-fill" style="width:${strPct}%"></div></div>
            <span class="sim-num">${hasStructure && m.structure_similarity != null ? m.structure_similarity.toFixed(3) : '—'}</span>
          </div>
        </td>
      </tr>
    `;
  });

  container.innerHTML = `
    <div class="chart-container">
      <div class="chart-title">Similarity Heatmap</div>
      <div id="chart-heatmap" style="width:100%;height:200px;"></div>
    </div>
    <div class="matches-table-wrap animate-in">
      <table class="matches-table">
        <thead>
          <tr>
            <th style="width:50px;">#</th>
            <th>Name</th>
            <th>Organism</th>
            <th style="width:100px;">Type</th>
            <th style="width:180px;">Embedding Sim</th>
            <th style="width:180px;">Structure Sim</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>
  `;

  setTimeout(() => renderSimilarityHeatmap('chart-heatmap', matches), 50);
}


// ── Section J: Structure Tab ────────────────────────────────────────

let currentViewStyle = 'cartoon';
let currentColorMode = 'Default';
let viewer3d = null;

function renderStructureTab(data) {
  const container = document.getElementById('tab-structure');
  const pdbString = data.pdb_string;

  if (!pdbString) {
    container.innerHTML = '<div class="info-box">Structure prediction did not return a result. Try again or check the ESMFold API.</div>';
    return;
  }

  const pocketRes = data.pocket_residues || [];
  const dangerRes = data.danger_residues || [];
  const alignedRegions = data.aligned_regions || [];

  container.innerHTML = `
    <div class="viewer-card">
      <div class="viewer-toolbar">
        <div class="viewer-toggle-group" id="view-style-group">
          <button class="viewer-toggle active" data-style="cartoon" onclick="setViewStyle('cartoon', this)">Cartoon</button>
          <button class="viewer-toggle" data-style="surface" onclick="setViewStyle('surface', this)">Surface</button>
          <button class="viewer-toggle" data-style="stick" onclick="setViewStyle('stick', this)">Stick</button>
        </div>
        <div class="viewer-toggle-group" id="color-mode-group">
          <button class="viewer-toggle active" data-color="Default" onclick="setColorMode('Default', this)">Default</button>
          <button class="viewer-toggle" data-color="pLDDT" onclick="setColorMode('pLDDT', this)">pLDDT</button>
          <button class="viewer-toggle" data-color="Risk Layers" onclick="setColorMode('Risk Layers', this)">Risk Layers</button>
        </div>
      </div>
      <div class="viewer-3d-container" id="viewer-3d"></div>
      <div class="viewer-legend" id="viewer-legend"></div>
    </div>
  `;

  currentViewStyle = 'cartoon';
  currentColorMode = 'Default';

  // Wait for 3Dmol to load
  waitFor3Dmol(() => {
    create3DViewer(pdbString, pocketRes, dangerRes, alignedRegions);
    updateLegend(pocketRes, dangerRes, alignedRegions);
  });
}

function waitFor3Dmol(callback, retries = 20) {
  if (window.$3Dmol) {
    callback();
  } else if (retries > 0) {
    setTimeout(() => waitFor3Dmol(callback, retries - 1), 250);
  }
}

function create3DViewer(pdbString, pocketRes, dangerRes, alignedRegions) {
  const container = document.getElementById('viewer-3d');
  if (!container) return;

  if (viewer3d) {
    viewer3d.clear();
  }

  viewer3d = $3Dmol.createViewer(container, { backgroundColor: 'white' });
  viewer3d.addModel(pdbString, 'pdb');

  const alignedRes = expandAlignedRegions(alignedRegions);
  const rep = currentViewStyle;

  if (currentColorMode === 'Risk Layers') {
    const baseStyle = {};
    baseStyle[rep] = { color: '#b0b0b0' };
    viewer3d.setStyle({ model: 0 }, baseStyle);

    if (alignedRes.length > 0) {
      const alignStyle = {};
      alignStyle[rep] = { color: '#fbbf24' };
      viewer3d.addStyle({ model: 0, resi: alignedRes }, alignStyle);
    }
  } else if (currentColorMode === 'pLDDT') {
    if (rep === 'cartoon') {
      viewer3d.setStyle({ model: 0 }, { cartoon: { colorscheme: { prop: 'b', gradient: 'roygb', min: 50, max: 100 } } });
    } else {
      const s = {};
      s[rep] = { colorscheme: { prop: 'b', gradient: 'roygb', min: 50, max: 100 } };
      viewer3d.setStyle({ model: 0 }, s);
    }
  } else {
    // Default
    if (rep === 'surface') {
      viewer3d.setStyle({ model: 0 }, { cartoon: { color: 'lightblue', opacity: 0.5 } });
      viewer3d.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.7, color: 'lightblue' }, { model: 0 });
    } else {
      const s = {};
      s[rep] = { color: 'lightblue' };
      viewer3d.setStyle({ model: 0 }, s);
    }
  }

  // Pocket residues: orange sticks
  if (pocketRes.length > 0) {
    viewer3d.addStyle({ model: 0, resi: pocketRes }, { stick: { color: 'orange', radius: 0.2 } });
  }

  // Danger residues: red sticks + transparent surface
  if (dangerRes.length > 0) {
    viewer3d.addStyle({ model: 0, resi: dangerRes }, { stick: { color: 'red', radius: 0.3 } });
    viewer3d.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.3, color: 'red' }, { model: 0, resi: dangerRes });
  }

  viewer3d.zoomTo();
  viewer3d.render();
  viewer3d.spin(false);
}

function setViewStyle(style, btn) {
  currentViewStyle = style;
  // Update toggle buttons
  btn.closest('.viewer-toggle-group').querySelectorAll('.viewer-toggle').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  // Re-render
  if (currentResult && currentResult.pdb_string) {
    create3DViewer(currentResult.pdb_string, currentResult.pocket_residues || [], currentResult.danger_residues || [], currentResult.aligned_regions || []);
  }
}

function setColorMode(mode, btn) {
  currentColorMode = mode;
  btn.closest('.viewer-toggle-group').querySelectorAll('.viewer-toggle').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (currentResult && currentResult.pdb_string) {
    create3DViewer(currentResult.pdb_string, currentResult.pocket_residues || [], currentResult.danger_residues || [], currentResult.aligned_regions || []);
    updateLegend(currentResult.pocket_residues || [], currentResult.danger_residues || [], currentResult.aligned_regions || []);
  }
}

function updateLegend(pocketRes, dangerRes, alignedRegions) {
  const legendEl = document.getElementById('viewer-legend');
  if (!legendEl) return;

  const alignedRes = expandAlignedRegions(alignedRegions);

  if (currentColorMode === 'Risk Layers') {
    const parts = ['<div class="legend-item"><div class="legend-dot" style="background:#b0b0b0;"></div> No structural match</div>'];
    if (alignedRes.length > 0) {
      parts.push(`<div class="legend-item"><div class="legend-dot" style="background:#fbbf24;"></div> Aligned to toxin (${alignedRes.length} residues)</div>`);
    }
    if (pocketRes.length > 0) {
      parts.push(`<div class="legend-item"><div class="legend-dot" style="background:orange;"></div> Active site pocket (${pocketRes.length})</div>`);
    }
    if (dangerRes.length > 0) {
      parts.push(`<div class="legend-item"><div class="legend-dot" style="background:red;"></div> Active site match (${dangerRes.length})</div>`);
    }
    legendEl.innerHTML = parts.join('');
  } else {
    const parts = ['<div class="legend-item"><div class="legend-dot" style="background:#94b8d8;"></div> Query protein</div>'];
    if (pocketRes.length > 0) {
      parts.push(`<div class="legend-item"><div class="legend-dot" style="background:orange;"></div> Active site pocket (${pocketRes.length})</div>`);
    }
    if (dangerRes.length > 0) {
      parts.push(`<div class="legend-item"><div class="legend-dot" style="background:red;"></div> Danger residues (${dangerRes.length})</div>`);
    }
    legendEl.innerHTML = parts.join('');
  }
}


// ── Section K: Score Breakdown Tab ──────────────────────────────────

function renderScoreBreakdownTab(data) {
  const container = document.getElementById('tab-scores');
  const factors = data.risk_factors || {};
  const riskScore = data.risk_score;
  const embSim = factors.max_embedding_similarity || 0;
  const structSim = factors.max_structure_similarity;
  const funcOverlap = factors.function_overlap || 0;

  let weightSet, weightNote;
  if (structSim != null) {
    weightSet = { Embedding: 0.50, Structure: 0.30, Function: 0.20 };
    weightNote = "Weights: embedding 0.50, structure 0.30, function 0.20";
  } else {
    weightSet = { Embedding: 0.65, Function: 0.35 };
    weightNote = "Weights: embedding 0.65, function 0.35 (no structure data available)";
  }

  const components = [
    { label: "Embedding Similarity", raw: embSim, weight: weightSet.Embedding || 0 },
  ];
  if (structSim != null) {
    components.push({ label: "Structure Similarity", raw: structSim, weight: weightSet.Structure || 0 });
  }

  const totalWeighted = components.reduce((s, c) => s + c.raw * c.weight, 0);
  const bonus = Math.max(0, riskScore - Math.min(1.0, totalWeighted));

  let scoreRows = '';
  components.forEach(c => {
    const pct = Math.round(c.raw * 100);
    const weighted = (c.raw * c.weight).toFixed(3);
    scoreRows += `
      <div class="score-row">
        <span class="score-label">${c.label}</span>
        <div class="score-track"><div class="score-fill" style="width:${pct}%"></div></div>
        <span class="score-num">${c.raw.toFixed(3)}</span>
      </div>
      <div class="score-row" style="margin-bottom:1.5rem;">
        <span class="score-meta">Weight: ${c.weight.toFixed(2)} &middot; Contribution: ${weighted}</span>
      </div>
    `;
  });

  if (bonus > 0.01) {
    const bonusPct = Math.round(bonus * 100);
    scoreRows += `
      <div class="score-row">
        <span class="score-label">Synergy Bonus</span>
        <div class="score-track"><div class="score-fill" style="width:${bonusPct}%; background:linear-gradient(90deg, #a78bfa, #8b5cf6);"></div></div>
        <span class="score-num">+${bonus.toFixed(3)}</span>
      </div>
      <div class="score-row" style="margin-bottom:1rem;">
        <span class="score-meta">Bonus for multiple high-confidence signals</span>
      </div>
    `;
  }

  const riskClass = riskScore >= 0.75 ? 'risk-high' : riskScore >= 0.45 ? 'risk-medium' : 'risk-low';

  container.innerHTML = `
    <div class="score-section animate-in">
      <div class="section-label" style="margin-bottom:1.25rem;">Component Scores</div>
      <p style="font-size:0.82rem; color:var(--text-secondary); margin-bottom:1.25rem;"><strong>Weight set:</strong> ${weightNote}</p>
      ${scoreRows}
      <div class="score-total">
        <span class="score-total-label">Final Risk Score</span>
        <span class="score-total-value ${riskClass}">${riskScore.toFixed(3)}</span>
      </div>
    </div>
  `;
}


// ── Section L: Function Tab ─────────────────────────────────────────

function renderFunctionTab(data) {
  const container = document.getElementById('tab-function');
  const fp = data.function_prediction;
  if (!fp) {
    container.innerHTML = '<div class="info-box">Function prediction not yet available. Polling InterPro...</div>';
    return;
  }

  const goTerms = fp.go_terms || [];
  const ecNumbers = fp.ec_numbers || [];

  let goHtml = '';
  if (goTerms.length > 0) {
    goHtml = '<div class="section-label">GO Terms</div><div class="func-grid">';
    goTerms.forEach(term => {
      const conf = parseFloat(term.confidence || 0);
      const confPct = Math.round(conf * 100);
      goHtml += `
        <div class="func-card">
          <div class="func-card-header">
            <span class="func-id">${escapeHtml(term.term || '')}</span>
            <span class="func-name">${escapeHtml(term.name || '')}</span>
          </div>
          <div class="func-bar"><div class="func-bar-fill" style="width:${confPct}%"></div></div>
          <div style="font-size:0.72rem; color:var(--text-tertiary); margin-top:0.25rem;">${conf.toFixed(2)}</div>
        </div>
      `;
    });
    goHtml += '</div>';
  }

  let ecHtml = '';
  if (ecNumbers.length > 0) {
    ecHtml = '<div class="section-label" style="margin-top:1.5rem;">EC Numbers</div><div class="func-grid">';
    ecNumbers.forEach(ec => {
      const conf = parseFloat(ec.confidence || 0);
      const confPct = Math.round(conf * 100);
      ecHtml += `
        <div class="func-card">
          <div class="func-card-header">
            <span class="func-id">EC ${escapeHtml(ec.number || '')}</span>
          </div>
          <div class="func-bar"><div class="func-bar-fill" style="width:${confPct}%"></div></div>
          <div style="font-size:0.72rem; color:var(--text-tertiary); margin-top:0.25rem;">${conf.toFixed(2)}</div>
        </div>
      `;
    });
    ecHtml += '</div>';
  }

  if (!goTerms.length && !ecNumbers.length) {
    container.innerHTML = '<div class="info-box">No GO terms or EC numbers predicted.</div>';
    return;
  }

  // Function overlap
  const topMatch = (data.top_matches || [])[0] || {};
  let overlapHtml = '';
  const queryTerms = new Set(goTerms.map(t => t.term || ''));
  const matchTerms = new Set((topMatch.go_terms || []).map(t => t.term || ''));
  const shared = [...queryTerms].filter(t => matchTerms.has(t));
  const queryOnly = [...queryTerms].filter(t => !matchTerms.has(t));
  const matchOnly = [...matchTerms].filter(t => !queryTerms.has(t));

  if (shared.length > 0 || queryOnly.length > 0 || matchOnly.length > 0) {
    overlapHtml = `
      <div class="section-divider"><span>Function Overlap</span></div>
      <div class="func-overlap">
        <div class="func-overlap-col">
          <h5>Query Only</h5>
          ${queryOnly.map(t => `<span class="func-id">${escapeHtml(t)}</span>`).join('') || '<span style="color:var(--text-tertiary);font-size:0.78rem;">None</span>'}
        </div>
        <div class="func-overlap-col">
          <h5>Shared</h5>
          ${shared.map(t => `<span class="func-id">${escapeHtml(t)}</span>`).join('') || '<span style="color:var(--text-tertiary);font-size:0.78rem;">None</span>'}
        </div>
        <div class="func-overlap-col">
          <h5>${escapeHtml(topMatch.name || 'Match')} Only</h5>
          ${matchOnly.map(t => `<span class="func-id">${escapeHtml(t)}</span>`).join('') || '<span style="color:var(--text-tertiary);font-size:0.78rem;">None</span>'}
        </div>
      </div>
    `;
  }

  container.innerHTML = `
    ${goHtml}
    ${ecHtml}
    <div class="chart-container" style="margin-top:1rem;">
      <div class="chart-title">Function Prediction Confidence</div>
      <div id="chart-func-bars" style="width:100%;"></div>
    </div>
    ${overlapHtml}
  `;

  setTimeout(() => renderFunctionBars('chart-func-bars', fp), 50);
}


// ── Section M: Explain Tab ──────────────────────────────────────────

function renderExplainTab(data) {
  const container = document.getElementById('tab-explain');
  const riskScore = data.risk_score;
  const riskLevel = data.risk_level;
  const factors = data.risk_factors || {};
  const topMatches = data.top_matches || [];
  const topMatch = topMatches[0] || {};
  const matchName = escapeHtml(topMatch.name || '');
  const matchOrganism = escapeHtml(topMatch.organism || '');
  const matchToxinType = topMatch.toxin_type || '';

  const embSim = factors.max_embedding_similarity || 0;
  const structSim = factors.max_structure_similarity;
  const funcOverlap = factors.function_overlap || 0;
  const activeSite = factors.active_site_overlap;
  const seqLength = data.sequence_length || 0;

  // Verdict
  let verdictClass, verdictIcon, verdictText;
  if (riskScore >= 0.75) {
    verdictClass = 'high';
    verdictIcon = '&#9888;&#65039;';
    verdictText = 'HIGH RISK: Strong similarity to known toxins';
  } else if (riskScore >= 0.45) {
    verdictClass = 'medium';
    verdictIcon = '&#9888;';
    verdictText = 'MODERATE RISK: Notable similarity to toxins';
  } else if (riskScore >= 0.2) {
    verdictClass = 'low';
    verdictIcon = '&#10003;';
    verdictText = 'LOW RISK: Minimal similarity';
  } else {
    verdictClass = 'low';
    verdictIcon = '&#10003;';
    verdictText = 'MINIMAL RISK: No significant similarity';
  }

  // Summary paragraph
  let summary;
  if (riskScore >= 0.75 && matchName) {
    summary = `This protein scored <strong>${riskScore.toFixed(3)}</strong> on the BioScreen risk scale, placing it in the <strong>high risk</strong> category. The submitted sequence shows strong similarity to <strong>${matchName}</strong>${matchToxinType && matchOrganism ? `, a ${escapeHtml(matchToxinType)} toxin from <em>${matchOrganism}</em>` : ''}. Multiple independent lines of evidence — including sequence embedding analysis, structural fold comparison, and functional annotation — converge to suggest that this protein could have biological activity consistent with a known dangerous agent. Immediate review by a biosafety expert is strongly recommended before any synthesis or experimental work proceeds.`;
  } else if (riskScore >= 0.45 && matchName) {
    summary = `This protein scored <strong>${riskScore.toFixed(3)}</strong> on the BioScreen risk scale, placing it in the <strong>moderate risk</strong> category. The screening pipeline detected notable similarity to <strong>${matchName}</strong>${matchToxinType && matchOrganism ? `, a ${escapeHtml(matchToxinType)} toxin from <em>${matchOrganism}</em>` : ''}. While the evidence is not conclusive enough to classify this as high risk, the signals are sufficient to warrant further investigation.`;
  } else if (riskScore >= 0.2) {
    summary = `This protein scored <strong>${riskScore.toFixed(3)}</strong> on the BioScreen risk scale, placing it in the <strong>low risk</strong> category. The screening pipeline detected only minimal similarity to known toxins.${matchName ? ` The closest match was <strong>${matchName}</strong>, but the similarity was not strong enough to raise concern.` : ''} Standard biosafety protocols should be sufficient.`;
  } else {
    summary = `This protein scored <strong>${riskScore.toFixed(3)}</strong> on the BioScreen risk scale, indicating <strong>minimal risk</strong>. No significant similarity to any known toxin was detected.`;
  }

  // Evidence blocks
  let evidenceHtml = '';

  // Embedding similarity
  let embDesc;
  if (embSim >= 0.97) {
    embDesc = `The ESM-2 protein language model found very high similarity (${embSim.toFixed(3)}) between this protein and known toxins. The sequence occupies nearly the same region of protein embedding space as established dangerous proteins.`;
  } else if (embSim >= 0.93) {
    embDesc = `The ESM-2 model detected moderate similarity (${embSim.toFixed(3)}) between this protein and known toxins. This may reflect shared folds or binding properties.`;
  } else if (embSim >= 0.85) {
    embDesc = `The ESM-2 model found only weak similarity (${embSim.toFixed(3)}). Scores in this range are typical of proteins sharing general structural features but lacking toxin-specific characteristics.`;
  } else {
    embDesc = `The ESM-2 model found no meaningful similarity (${embSim.toFixed(3)}) to known toxins.`;
  }
  evidenceHtml += `
    <div class="evidence-block">
      <div class="evidence-block-header">
        <span class="evidence-label">Sequence Embedding Similarity</span>
        <span class="evidence-value" style="color:${embSim >= 0.93 ? COLOR_RED : embSim >= 0.85 ? COLOR_ORANGE : COLOR_GREEN};">${embSim.toFixed(3)}</span>
      </div>
      <div class="evidence-text">${embDesc}</div>
    </div>
  `;

  // Structural similarity
  if (structSim != null) {
    let structDesc;
    if (structSim >= 0.8) {
      structDesc = `Foldseek structural alignment found very high 3D fold similarity (TM-score ${structSim.toFixed(3)}). A TM-score above 0.8 strongly suggests these proteins share the same overall fold topology. This is the key gap BioScreen addresses: AI-designed proteins can adopt toxin-like folds while evading traditional BLAST-based screening.`;
    } else if (structSim >= 0.5) {
      structDesc = `Foldseek detected moderate structural similarity (TM-score ${structSim.toFixed(3)}). The proteins may share common structural motifs or sub-domains without identical overall folds.`;
    } else {
      structDesc = `Foldseek found low 3D fold similarity (TM-score ${structSim.toFixed(3)}). These proteins likely have distinct overall folds.`;
    }
    evidenceHtml += `
      <div class="evidence-block">
        <div class="evidence-block-header">
          <span class="evidence-label">Structural Similarity</span>
          <span class="evidence-value" style="color:${structSim >= 0.8 ? COLOR_RED : structSim >= 0.5 ? COLOR_ORANGE : COLOR_GREEN};">${structSim.toFixed(3)}</span>
        </div>
        <div class="evidence-text">${structDesc}</div>
      </div>
    `;

    // Sequence-structure divergence
    const seqIdentity = topMatch.sequence_identity;
    if (seqIdentity != null && seqIdentity < 0.3 && structSim > 0.5) {
      evidenceHtml += `
        <div class="recommend-card" style="border-left-color:${COLOR_ORANGE};">
          <strong>Sequence-Structure Divergence:</strong> Low sequence identity (${(seqIdentity * 100).toFixed(0)}%) with moderate-to-high structural similarity (TM-score ${structSim.toFixed(3)}) — a hallmark of AI-designed proteins that can mimic toxin folds while appearing completely different at the sequence level.
        </div>
      `;
    }
  } else {
    evidenceHtml += `
      <div class="evidence-block">
        <div class="evidence-block-header">
          <span class="evidence-label">Structural Similarity</span>
          <span class="evidence-value">N/A</span>
        </div>
        <div class="evidence-text">Structure prediction was not available. The risk assessment relies solely on sequence embeddings and functional annotations.</div>
      </div>
    `;
  }

  // Active site
  if (activeSite != null) {
    let activeDesc;
    if (activeSite >= 0.7) {
      activeDesc = `High similarity (${activeSite.toFixed(3)}) between this protein's active site and the matched toxin's functional site. This is the strongest indicator that the protein could have functional toxicity.`;
    } else if (activeSite >= 0.4) {
      activeDesc = `Moderate geometric similarity (${activeSite.toFixed(3)}) between active sites. There is enough spatial resemblance to suggest partial conservation of functional site architecture.`;
    } else {
      activeDesc = `Low active site similarity (${activeSite.toFixed(3)}). The lack of active site conservation reduces the likelihood of replicating toxin function.`;
    }
    evidenceHtml += `
      <div class="evidence-block">
        <div class="evidence-block-header">
          <span class="evidence-label">Active Site Similarity</span>
          <span class="evidence-value" style="color:${activeSite >= 0.7 ? COLOR_RED : activeSite >= 0.4 ? COLOR_ORANGE : 'var(--text-secondary)'};">${activeSite.toFixed(3)}</span>
        </div>
        <div class="evidence-text">${activeDesc}</div>
      </div>
    `;
  }

  // Function overlap
  if (data.function_prediction) {
    let funcDesc;
    if (funcOverlap >= 0.6) {
      funcDesc = `Significant overlap (${funcOverlap.toFixed(3)}) between predicted GO terms/EC numbers and those of the matched toxin, corroborating structural and sequence-based evidence.`;
    } else if (funcOverlap >= 0.3) {
      funcDesc = `Moderate functional overlap (${funcOverlap.toFixed(3)}). Some shared biological function, though this could reflect broadly conserved activities.`;
    } else if (funcOverlap > 0) {
      funcDesc = `Minimal functional overlap (${funcOverlap.toFixed(3)}). The protein appears to have a different predicted biological role.`;
    } else {
      funcDesc = `No functional overlap detected (0.000).`;
    }
    evidenceHtml += `
      <div class="evidence-block">
        <div class="evidence-block-header">
          <span class="evidence-label">Functional Annotation Overlap</span>
          <span class="evidence-value" style="color:${funcOverlap >= 0.6 ? COLOR_RED : funcOverlap >= 0.3 ? COLOR_ORANGE : 'var(--text-secondary)'};">${funcOverlap.toFixed(3)}</span>
        </div>
        <div class="evidence-text">${funcDesc}</div>
      </div>
    `;
  }

  // Short sequence note
  let shortSeqNote = '';
  if (seqLength && seqLength < 50) {
    shortSeqNote = `
      <div class="recommend-card">
        <strong>Note on sequence length:</strong> This protein is only ${seqLength} amino acids long. Short peptides (under 50 residues) tend to cluster in ESM-2 embedding space regardless of function, which can inflate similarity scores.
      </div>
    `;
  }

  // Recommendation
  let recommendation = '';
  if (riskScore >= 0.75) {
    recommendation = `
      <div class="recommend-card">
        <strong>Recommendation:</strong> Immediate review by a biosafety expert is strongly recommended. This protein should not proceed to synthesis or experimental work without thorough manual evaluation.
      </div>
    `;
  } else if (riskScore >= 0.45) {
    recommendation = `
      <div class="recommend-card">
        <strong>Recommendation:</strong> A biosafety review is recommended before proceeding. Enhanced monitoring protocols should be applied if this protein moves forward.
      </div>
    `;
  } else if (riskScore >= 0.3) {
    recommendation = `
      <div class="recommend-card">
        <strong>Recommendation:</strong> Standard biosafety protocols should be sufficient. Routine screening documentation should be maintained.
      </div>
    `;
  }

  // Session monitoring
  const anomalyScore = factors.session_anomaly_score || 0;
  let sessionHtml = '';
  if (anomalyScore > 0) {
    let sessionClass, sessionText;
    if (anomalyScore > 0.5) {
      sessionClass = 'alert-error';
      sessionText = `Session anomaly score: ${anomalyScore.toFixed(2)} — A convergent optimization pattern has been detected.`;
    } else if (anomalyScore > 0.3) {
      sessionClass = 'alert-warning';
      sessionText = `Session anomaly score: ${anomalyScore.toFixed(2)} — Elevated activity detected.`;
    } else {
      sessionClass = 'alert-info';
      sessionText = `Session anomaly score: ${anomalyScore.toFixed(2)} (normal)`;
    }
    sessionHtml = `
      <div class="section-divider"><span>Session Monitoring</span></div>
      <div class="alert-banner ${sessionClass}">${sessionText}</div>
    `;
  }

  // Warnings
  let warningsHtml = '';
  const warnings = data.warnings || [];
  if (warnings.length > 0) {
    warningsHtml = '<div class="section-divider"><span>Warnings</span></div>';
    warnings.forEach(w => {
      warningsHtml += `<div class="alert-banner alert-warning">${escapeHtml(w)}</div>`;
    });
  }

  container.innerHTML = `
    <div class="verdict-card ${verdictClass} animate-in">
      <span class="verdict-icon">${verdictIcon}</span>
      <span>${verdictText}</span>
    </div>

    <div class="evidence-block" style="margin-bottom:1.5rem;">
      <div class="evidence-label" style="margin-bottom:0.5rem;">What does this mean?</div>
      <div class="evidence-text">${summary}</div>
    </div>

    <div class="section-divider"><span>Evidence Breakdown</span></div>

    <div class="evidence-section">
      ${evidenceHtml}
    </div>

    ${shortSeqNote}
    ${recommendation}
    ${sessionHtml}
    ${warningsHtml}
  `;
}


// ── Section N: Session Analysis ─────────────────────────────────────

let sessionInputMode = 'paste';

function setSessionInputMode(mode) {
  sessionInputMode = mode;
  document.getElementById('toggle-paste').classList.toggle('active', mode === 'paste');
  document.getElementById('toggle-demo').classList.toggle('active', mode === 'demo');
  document.getElementById('session-paste-mode').classList.toggle('view-hidden', mode !== 'paste');
  document.getElementById('session-demo-mode').classList.toggle('view-hidden', mode !== 'demo');
}

function parseMultiFasta(raw) {
  const sequences = [];
  let currentLabel = null;
  let currentSeqLines = [];

  for (const line of raw.split('\n')) {
    const stripped = line.trim();
    if (!stripped) {
      if (currentSeqLines.length) {
        sequences.push([currentLabel || `Sequence ${sequences.length + 1}`, currentSeqLines.join('')]);
        currentLabel = null;
        currentSeqLines = [];
      }
      continue;
    }
    if (stripped.startsWith('>')) {
      if (currentSeqLines.length) {
        sequences.push([currentLabel || `Sequence ${sequences.length + 1}`, currentSeqLines.join('')]);
        currentSeqLines = [];
      }
      currentLabel = stripped.substring(1).trim() || `Sequence ${sequences.length + 1}`;
    } else {
      currentSeqLines.push(stripped);
    }
  }
  if (currentSeqLines.length) {
    sequences.push([currentLabel || `Sequence ${sequences.length + 1}`, currentSeqLines.join('')]);
  }
  return sequences;
}

async function runSessionAnalysis() {
  let sequences;
  if (sessionInputMode === 'paste') {
    const raw = document.getElementById('session-fasta-input').value.trim();
    if (!raw) { alert('Please paste multi-FASTA sequences.'); return; }
    sequences = parseMultiFasta(raw);
    if (!sequences.length) { alert('No valid sequences found.'); return; }
  } else {
    sequences = DEMO_CONVERGENCE_SERIES;
  }

  const topK = parseInt(document.getElementById('session-topk').value) || 5;
  const sid = crypto.randomUUID();

  // Show progress
  const progressEl = document.getElementById('session-progress');
  const progressFill = document.getElementById('session-progress-fill');
  const progressText = document.getElementById('session-progress-text');
  progressEl.classList.remove('view-hidden');
  document.getElementById('session-results').classList.add('view-hidden');

  const btn = document.getElementById('btn-run-session');
  btn.disabled = true;

  const results = [];

  for (let i = 0; i < sequences.length; i++) {
    const [label, seq] = sequences[i];
    const pct = Math.round((i / sequences.length) * 100);
    progressFill.style.width = `${pct}%`;
    progressText.textContent = `Screening ${i + 1}/${sequences.length}: ${label.substring(0, 60)}...`;

    try {
      const data = await screenSequence(seq, label, topK, sid);
      results.push({ label, success: true, data });
    } catch (err) {
      results.push({ label, success: false, error: err.message });
    }
  }

  progressFill.style.width = '100%';
  progressText.textContent = 'Done!';

  // Fetch alerts
  const alerts = await getSessionAlerts(sid);

  btn.disabled = false;

  renderSessionResults(results, alerts);
}

function renderSessionResults(results, alerts) {
  const container = document.getElementById('session-results');
  container.classList.remove('view-hidden');

  let html = '';

  // Alert banner
  if (alerts) {
    const anomalyScore = alerts.anomaly_score || 0;
    const explanation = alerts.explanation || '';
    if (anomalyScore > 0.5) {
      html += `<div class="alert-banner alert-error"><strong>Anomaly detected</strong> (score ${anomalyScore.toFixed(2)}): ${escapeHtml(explanation)}</div>`;
    } else if (anomalyScore > 0.3) {
      html += `<div class="alert-banner alert-warning"><strong>Elevated anomaly score</strong> (${anomalyScore.toFixed(2)}): ${escapeHtml(explanation)}</div>`;
    }
  }

  // Summary table
  const rows = [];
  results.forEach((res, idx) => {
    if (res.success && res.data) {
      const d = res.data;
      const riskClass = d.risk_level === 'HIGH' ? 'high' : d.risk_level === 'MEDIUM' ? 'medium' : 'low';
      const topMatchName = (d.top_matches || [])[0]?.name || '—';
      rows.push({
        idx: idx + 1,
        label: res.label,
        score: d.risk_score,
        level: d.risk_level,
        levelClass: riskClass,
        topMatch: topMatchName,
      });
    } else {
      rows.push({
        idx: idx + 1,
        label: res.label,
        score: null,
        level: 'error',
        levelClass: '',
        topMatch: res.error || 'Failed',
      });
    }
  });

  html += `
    <div class="section-label" style="margin-top:1.5rem;">Results Summary</div>
    <div class="matches-table-wrap">
      <table class="summary-table">
        <thead>
          <tr><th>#</th><th>Sequence</th><th>Risk Score</th><th>Risk Level</th><th>Top Match</th></tr>
        </thead>
        <tbody>
          ${rows.map(r => `
            <tr>
              <td style="font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text-tertiary);">${r.idx}</td>
              <td>${escapeHtml(r.label.substring(0, 50))}</td>
              <td style="font-family:'JetBrains Mono',monospace;font-weight:600;">${r.score != null ? r.score.toFixed(3) : '—'}</td>
              <td>${r.level !== 'error' ? `<span class="risk-badge ${r.levelClass}">${r.level}</span>` : '<span style="color:var(--risk-high);">Error</span>'}</td>
              <td>${escapeHtml(r.topMatch)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;

  // Risk trend chart
  const scores = rows.filter(r => r.score != null).map(r => r.score);
  if (scores.length > 1) {
    html += `
      <div class="chart-container">
        <div class="chart-title">Risk Score Trend</div>
        <div id="chart-risk-trend" style="width:100%;height:250px;"></div>
      </div>
    `;
  }

  // Session monitoring metrics
  if (alerts) {
    const convergence = alerts.convergence || {};
    const perturbation = alerts.perturbation || {};

    html += `
      <div class="section-label" style="margin-top:1.5rem;">Session Monitoring</div>
      <div class="metric-grid">
        <div class="metric-card">
          <h4>Convergence Detector</h4>
          <div class="metric-item">
            <span class="metric-item-label">Status</span>
            <span class="metric-item-value" style="color:${convergence.is_flagged ? COLOR_RED : COLOR_GREEN};">${convergence.is_flagged ? 'FLAGGED' : 'Normal'}</span>
          </div>
          <div class="metric-item">
            <span class="metric-item-label">Mean Similarity</span>
            <span class="metric-item-value">${(convergence.mean_similarity || 0).toFixed(4)}</span>
          </div>
          <div class="metric-item">
            <span class="metric-item-label">Similarity Trend</span>
            <span class="metric-item-value">${(convergence.similarity_trend || 0).toFixed(4)}</span>
          </div>
          <div class="metric-item">
            <span class="metric-item-label">Window Size</span>
            <span class="metric-item-value">${convergence.window_size || 0}</span>
          </div>
        </div>
        <div class="metric-card">
          <h4>Perturbation Detector</h4>
          <div class="metric-item">
            <span class="metric-item-label">Status</span>
            <span class="metric-item-value" style="color:${perturbation.is_flagged ? COLOR_RED : COLOR_GREEN};">${perturbation.is_flagged ? 'FLAGGED' : 'Normal'}</span>
          </div>
          <div class="metric-item">
            <span class="metric-item-label">Cluster Count</span>
            <span class="metric-item-value">${perturbation.cluster_count || 0}</span>
          </div>
          <div class="metric-item">
            <span class="metric-item-label">Max Cluster Size</span>
            <span class="metric-item-value">${perturbation.max_cluster_size || 0}</span>
          </div>
          <div class="metric-item">
            <span class="metric-item-label">High-Sim Pairs</span>
            <span class="metric-item-value">${(perturbation.high_sim_pairs || []).length}</span>
          </div>
        </div>
      </div>
    `;
  }

  // Per-sequence accordions
  html += '<div class="section-label" style="margin-top:1.5rem;">Per-Sequence Detail</div>';
  results.forEach((res, idx) => {
    const riskClass = res.success && res.data ? (res.data.risk_level === 'HIGH' ? 'high' : res.data.risk_level === 'MEDIUM' ? 'medium' : 'low') : '';
    const score = res.success && res.data ? res.data.risk_score.toFixed(3) : '—';
    const level = res.success && res.data ? res.data.risk_level : 'error';

    html += `
      <div class="accordion" id="accordion-${idx}">
        <div class="accordion-header" onclick="toggleAccordion(${idx})">
          <span>${escapeHtml(res.label)}</span>
          <div class="accordion-meta">
            ${level !== 'error' ? `<span class="risk-badge ${riskClass}">${level}</span>` : ''}
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:600;">${score}</span>
            <span class="accordion-arrow">&#9660;</span>
          </div>
        </div>
        <div class="accordion-body" id="accordion-body-${idx}">
          ${res.success && res.data ? renderMiniResults(res.data, idx) : `<div class="error-box">${escapeHtml(res.error || 'Screening failed.')}</div>`}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;

  // Render risk trend chart
  if (scores.length > 1) {
    setTimeout(() => {
      const trace = {
        x: scores.map((_, i) => i + 1),
        y: scores,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLOR_RED, width: 2 },
        marker: { size: 8, color: scores.map(s => s >= 0.75 ? COLOR_RED : s >= 0.45 ? COLOR_ORANGE : COLOR_GREEN) },
      };
      const layout = {
        ...PLOTLY_LAYOUT_BASE,
        xaxis: { title: 'Sequence #', dtick: 1 },
        yaxis: { range: [0, 1], title: 'Risk Score' },
        margin: { l: 50, r: 20, t: 20, b: 50 },
        shapes: [
          { type: 'line', x0: 0.5, x1: scores.length + 0.5, y0: 0.75, y1: 0.75, line: { color: COLOR_RED, width: 1, dash: 'dash' } },
          { type: 'line', x0: 0.5, x1: scores.length + 0.5, y0: 0.45, y1: 0.45, line: { color: COLOR_ORANGE, width: 1, dash: 'dash' } },
        ],
      };
      Plotly.newPlot('chart-risk-trend', [trace], layout, PLOTLY_CONFIG);
    }, 100);
  }
}

function toggleAccordion(idx) {
  const el = document.getElementById(`accordion-${idx}`);
  const wasOpen = el.classList.contains('open');
  el.classList.toggle('open');

  // Render mini charts if just opened and not yet rendered
  if (!wasOpen) {
    const radarEl = document.getElementById(`seq-${idx}-radar`);
    if (radarEl && !radarEl.data) {
      const res = null; // We'll find the data from the rendered HTML
      // Charts are rendered inline on first accordion open
      renderMiniCharts(idx);
    }
  }
}

function renderMiniResults(data, idx) {
  const prefix = `seq-${idx}`;
  const riskScore = data.risk_score;
  const riskLevel = data.risk_level;
  const riskClass = riskLevel === 'HIGH' ? 'high' : riskLevel === 'MEDIUM' ? 'medium' : 'low';
  const factors = data.risk_factors || {};
  const topMatch = (data.top_matches || [])[0] || {};

  // Mini tabs
  const tabs = ['overview', 'matches', 'scores', 'explain'];

  return `
    <div class="mini-tab-bar">
      ${tabs.map((t, i) => `<button class="mini-tab-item ${i === 0 ? 'active' : ''}" onclick="showMiniTab('${prefix}', '${t}', this)">${t.charAt(0).toUpperCase() + t.slice(1)}</button>`).join('')}
    </div>

    <div id="${prefix}-overview" data-mini-group="${prefix}">
      <div class="cards-row" style="grid-template-columns:1fr 1fr;">
        <div class="summary-card card-risk" style="padding:1rem;">
          <div class="card-label">Risk Score</div>
          <div class="card-value risk-${riskClass}" style="font-size:1.8rem;">${riskScore.toFixed(3)}</div>
          <div class="risk-bar"><div class="risk-bar-fill ${riskClass}" style="width:${Math.round(riskScore * 100)}%"></div></div>
          <span class="risk-badge ${riskClass}">${riskLevel}</span>
        </div>
        <div class="summary-card card-match" style="padding:1rem;">
          <div class="card-label">Top Match</div>
          <div class="match-name" style="font-size:0.9rem;">${escapeHtml(topMatch.name || 'No match')}</div>
          <div class="match-organism">${escapeHtml(topMatch.organism || '')}</div>
        </div>
      </div>
      <div class="chart-row">
        <div id="${prefix}-radar" style="width:100%;height:250px;" data-idx="${idx}"></div>
        <div id="${prefix}-donut" style="width:100%;height:250px;" data-idx="${idx}"></div>
      </div>
    </div>

    <div id="${prefix}-matches" data-mini-group="${prefix}" class="view-hidden">
      ${renderMiniMatchesTable(data)}
    </div>

    <div id="${prefix}-scores" data-mini-group="${prefix}" class="view-hidden">
      ${renderMiniScoreBreakdown(data)}
    </div>

    <div id="${prefix}-explain" data-mini-group="${prefix}" class="view-hidden">
      ${renderMiniExplain(data)}
    </div>
  `;
}

function renderMiniCharts(idx) {
  // Find the data - we stored it in results
  const container = document.getElementById('session-results');
  if (!container) return;

  // We need to re-find the data. Let's store results globally.
  if (!window._sessionResults) return;
  const res = window._sessionResults[idx];
  if (!res || !res.success || !res.data) return;

  const factors = res.data.risk_factors || {};
  const prefix = `seq-${idx}`;

  const radarEl = document.getElementById(`${prefix}-radar`);
  const donutEl = document.getElementById(`${prefix}-donut`);

  if (radarEl && !radarEl.data) {
    renderRadarChart(`${prefix}-radar`, factors);
  }
  if (donutEl && !donutEl.data) {
    renderDonutChart(`${prefix}-donut`, factors, res.data.risk_score);
  }
}

function renderMiniMatchesTable(data) {
  const matches = data.top_matches || [];
  if (!matches.length) return '<div class="info-box">No matches.</div>';

  let rows = '';
  matches.forEach((m, i) => {
    const embPct = Math.round((m.embedding_similarity || 0) * 100);
    rows += `
      <tr>
        <td class="rank">${i + 1}</td>
        <td><strong>${escapeHtml(m.name || '')}</strong></td>
        <td>
          <div class="sim-bar-cell">
            <div class="sim-bar"><div class="sim-bar-fill" style="width:${embPct}%"></div></div>
            <span class="sim-num">${(m.embedding_similarity || 0).toFixed(3)}</span>
          </div>
        </td>
      </tr>
    `;
  });

  return `
    <div class="matches-table-wrap">
      <table class="matches-table">
        <thead><tr><th>#</th><th>Name</th><th>Embedding Sim</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderMiniScoreBreakdown(data) {
  const factors = data.risk_factors || {};
  const riskScore = data.risk_score;
  const embSim = factors.max_embedding_similarity || 0;
  const structSim = factors.max_structure_similarity;
  const funcOverlap = factors.function_overlap || 0;

  const components = [{ label: "Embedding", raw: embSim, weight: structSim != null ? 0.50 : 0.65 }];
  if (structSim != null) components.push({ label: "Structure", raw: structSim, weight: 0.30 });
  components.push({ label: "Function", raw: funcOverlap, weight: structSim != null ? 0.20 : 0.35 });

  let html = '<div class="score-section" style="padding:1rem;">';
  components.forEach(c => {
    const pct = Math.round(c.raw * 100);
    html += `
      <div class="score-row" style="margin-bottom:0.5rem;">
        <span class="score-label" style="font-size:0.78rem;">${c.label}</span>
        <div class="score-track" style="height:20px;"><div class="score-fill" style="width:${pct}%"></div></div>
        <span class="score-num" style="font-size:0.78rem;">${c.raw.toFixed(3)}</span>
      </div>
    `;
  });
  const riskClass = riskScore >= 0.75 ? 'risk-high' : riskScore >= 0.45 ? 'risk-medium' : 'risk-low';
  html += `
    <div class="score-total" style="padding-top:0.5rem;">
      <span class="score-total-label" style="font-size:0.78rem;">Final</span>
      <span class="score-total-value ${riskClass}" style="font-size:1.2rem;">${riskScore.toFixed(3)}</span>
    </div>
  </div>`;
  return html;
}

function renderMiniExplain(data) {
  const riskScore = data.risk_score;
  const riskLevel = data.risk_level;
  const verdictClass = riskScore >= 0.75 ? 'high' : riskScore >= 0.45 ? 'medium' : 'low';
  const verdictText = riskScore >= 0.75 ? 'HIGH RISK' : riskScore >= 0.45 ? 'MODERATE RISK' : 'LOW RISK';
  const topMatch = (data.top_matches || [])[0] || {};

  return `
    <div class="verdict-card ${verdictClass}" style="padding:0.85rem 1rem; font-size:0.85rem;">
      <span>${verdictText}: ${riskScore.toFixed(3)}</span>
    </div>
    <div class="evidence-text" style="font-size:0.8rem;">
      ${topMatch.name ? `Closest match: <strong>${escapeHtml(topMatch.name)}</strong> (${escapeHtml(topMatch.organism || '')})` : 'No significant matches found.'}
    </div>
  `;
}


// ── Section O: Utilities ────────────────────────────────────────────

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function expandAlignedRegions(regions) {
  const residues = new Set();
  for (const region of (regions || [])) {
    if (region.length === 2) {
      for (let i = region[0]; i <= region[1]; i++) {
        residues.add(i);
      }
    }
  }
  return [...residues].sort((a, b) => a - b);
}

function formatScore(num, decimals = 3) {
  return (num || 0).toFixed(decimals);
}

function copyResultJson() {
  if (!currentResult) return;
  const json = JSON.stringify(currentResult, null, 2);
  navigator.clipboard.writeText(json).then(() => {
    const btn = document.getElementById('btn-copy-json');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  }).catch(() => {
    // Fallback: show in a modal-like display
    const pre = document.createElement('pre');
    pre.textContent = json;
    pre.style.cssText = 'max-height:400px;overflow:auto;font-size:0.75rem;background:var(--bg-input);padding:1rem;border-radius:var(--radius-md);margin-top:1rem;';
    const actions = document.querySelector('.bottom-actions');
    if (actions.querySelector('pre')) actions.querySelector('pre').remove();
    actions.appendChild(pre);
  });
}

function updateApiStatus(health) {
  const pill = document.getElementById('api-status');
  const text = document.getElementById('api-status-text');
  if (health && health.status === 'ok') {
    pill.classList.remove('offline');
    text.textContent = 'API Connected';
  } else {
    pill.classList.add('offline');
    text.textContent = 'API Offline';
  }
}


// ── Section P: Initialization ───────────────────────────────────────

// Store session results globally for accordion chart rendering
window._sessionResults = null;

// Override renderSessionResults to store results
const _origRenderSession = renderSessionResults;
renderSessionResults = function(results, alerts) {
  window._sessionResults = results;
  _origRenderSession(results, alerts);
};

document.addEventListener('DOMContentLoaded', async () => {
  initSingleScreen();

  const health = await checkHealth();
  updateApiStatus(health);

  // Handle hash-based routing
  if (location.hash === '#session-analysis') {
    showView('session-analysis');
  }
});
