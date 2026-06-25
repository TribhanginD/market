const state = {
  snapshot: null,
  history: null,
  activeView: "overview",
};

const $ = (id) => document.getElementById(id);

const BENCHMARK_LABEL = "Benchmark Index";

const fmtMoney = (value) => {
  const v = Number(value || 0);
  if (Math.abs(v) >= 1e7) return `${(v / 1e7).toFixed(2)}Cr`;
  if (Math.abs(v) >= 1e5) return `${(v / 1e5).toFixed(2)}L`;
  return v.toLocaleString("en-IN", { maximumFractionDigits: 0 });
};

const fmtPct = (value, signed = false) => {
  const v = Number(value || 0);
  const sign = signed && v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
};

const fmtPrice = (value) => {
  const v = Number(value || 0);
  if (!v) return "n/a";
  return v.toLocaleString("en-IN", { maximumFractionDigits: 2 });
};

const fmtInt = (value) => Number(value || 0).toLocaleString("en-IN");

const esc = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const short = (value, limit = 120) => {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1).trim()}…` : text;
};

const actionBadge = (action) => {
  const a = String(action || "HOLD").toUpperCase();
  return `<span class="badge ${a.toLowerCase()}">${esc(a)}</span>`;
};

const plotLayout = (extra = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#d7e3ee", family: "Inter, sans-serif" },
  margin: { l: 48, r: 20, t: 20, b: 42 },
  xaxis: { gridcolor: "#1d2834", zerolinecolor: "#253241" },
  yaxis: { gridcolor: "#1d2834", zerolinecolor: "#253241" },
  legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right" },
  ...extra,
});

const plotConfig = { displayModeBar: false, responsive: true };

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadSnapshot() {
  const run = $("runSelect").value;
  const live = $("liveToggle").checked ? "1" : "0";
  const url = `/api/snapshot?live=${live}${run ? `&run=${encodeURIComponent(run)}` : ""}`;
  state.snapshot = await fetchJson(url);
  hydrateRunSelect();
  renderAll();
  await loadHistory();
}

async function loadHistory() {
  const symbols = state.snapshot.positions.slice(0, 8).map((p) => p.symbol).join(",");
  const period = $("periodSelect").value;
  state.history = await fetchJson(`/api/history?period=${encodeURIComponent(period)}&symbols=${encodeURIComponent(symbols)}`);
  renderCharts();
}

function hydrateRunSelect() {
  const select = $("runSelect");
  const current = select.value || state.snapshot.selected_run;
  select.innerHTML = state.snapshot.runs
    .map((run) => `<option value="${esc(run)}"${run === current ? " selected" : ""}>${esc(run)}</option>`)
    .join("");
}

function updateClock() {
  $("clockTime").textContent = new Date().toLocaleString();
}

function table(headers, rows, opts = {}) {
  if (!rows.length) return `<div class="empty">No data available.</div>`;
  return `
    <div class="table-wrap">
      <table>
        <thead><tr>${headers.map((h) => `<th class="${h.num ? "num" : ""}">${esc(h.label)}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows
            .map(
              (row) => `<tr>${headers
                .map((h) => `<td class="${h.num ? "num" : ""}">${opts.raw?.includes(h.key) ? row[h.key] ?? "" : esc(row[h.key] ?? "")}</td>`)
                .join("")}</tr>`,
            )
            .join("")}
        </tbody>
      </table>
    </div>`;
}

function renderAll() {
  const snap = state.snapshot;
  $("statusLine").textContent = `Run ${snap.selected_run || "n/a"} · last portfolio update ${snap.metrics.last_run || "never"} · ${snap.positions.length} holdings`;
  $("quoteTime").textContent = snap.metrics.quote_time ? `Quote ${snap.metrics.quote_time}` : "Quotes cached/offline";
  $("rulesBox").innerHTML = `
    <b>Portfolio rules</b><br>
    Capital: ${fmtMoney(snap.rules.capital_inr)}<br>
    Positions: ${snap.rules.min_positions}-${snap.rules.max_positions}<br>
    Max position: ${snap.rules.max_position_pct.toFixed(0)}%<br>
    Max sector: ${snap.rules.max_sector_pct.toFixed(0)}%<br>
    Swap hurdle: ${snap.rules.swap_hurdle_pct.toFixed(0)}% EV<br>
    Turnover cap: ${snap.rules.turnover_cap_pct.toFixed(0)}%<br>
    Benchmark: ${esc(BENCHMARK_LABEL)}
  `;
  renderRunRibbon();
  renderKpis();
  renderBrief();
  renderHero();
  renderFindings();
  renderStageBurn();
  renderOverviewTables();
  renderRunLibrary();
  renderMarket();
  renderFilters();
  renderHoldings();
  renderDecisions();
  renderRebalance();
  renderScenarios();
  renderMonitor();
}

function renderRunRibbon() {
  const selected = state.snapshot.selected_run_summary || {};
  const chips = [
    `${selected.duration_human || "n/a"} runtime`,
    `${fmtInt(selected.token_total || 0)} total tokens`,
    `${selected.positive_ev_count || 0} positive EV names`,
    `${selected.trades || 0} actions taken`,
    `${selected.warnings || 0} validation warnings`,
  ];
  $("runRibbon").innerHTML = chips.map((chip) => `<span class="ribbon-chip">${esc(chip)}</span>`).join("");
}

function renderBrief() {
  const snap = state.snapshot;
  const positions = [...snap.positions];
  const trades = [...(snap.selected_trades || [])];
  const stage5 = snap.stage5 || {};
  const summary = stage5.rebalance_summary || {};
  const topEv = positions.sort((a, b) => Number(b.ev_12m_pct || 0) - Number(a.ev_12m_pct || 0)).slice(0, 3);
  const latestTrades = trades.slice().reverse().slice(0, 4);
  const buys = Number(summary.buys || latestTrades.filter((t) => t.action === "BUY").length || 0);
  const sells = Number(summary.sells || latestTrades.filter((t) => t.action === "SELL").length || 0);
  const turnover = Number(summary.turnover_pct || 0);
  const stance = snap.metrics.urgent_alerts
    ? "Risk review required"
    : sells > 0
      ? "Active rebalance"
      : buys > 0
        ? "Constructive allocation"
        : "Hold and monitor";
  const stanceClass = snap.metrics.urgent_alerts ? "down" : sells > 0 ? "warn" : "up";
  const evidence = topEv
    .map((p) => `${p.symbol} leads modeled upside at ${fmtPct(p.ev_12m_pct, true)} with ${fmtPct(p.target_pct)} target weight`)
    .join(". ");
  const note = stage5.rebalance_notes || "The current run records model decisions, target weights, scenario returns, and thesis-monitor signals for audit.";

  $("briefPanel").innerHTML = `
    <article class="brief-main">
      <div class="panel-head">
        <div>
          <p class="eyebrow">Investment Committee Brief</p>
          <h3>Model stance: <span class="${stanceClass}">${esc(stance)}</span></h3>
        </div>
        <span class="pill">${esc(snap.selected_run || "latest")}</span>
      </div>
      <p>${esc(note)}</p>
      <p>${esc(evidence || "Scenario evidence appears after stage 3 output is available.")}</p>
      <div class="brief-grid">
        <div class="brief-cell"><span>Expected return</span><strong>${fmtPct(snap.metrics.weighted_ev_pct, true)}</strong></div>
        <div class="brief-cell"><span>Decision mix</span><strong>${buys} buy / ${sells} sell</strong></div>
        <div class="brief-cell"><span>Turnover</span><strong>${fmtPct(turnover)}</strong></div>
      </div>
    </article>
    <article class="brief-side">
      <div class="panel-head">
        <h3>Latest Model Moves</h3>
        <span class="pill">audit trail</span>
      </div>
      <div class="brief-list">
        ${
          latestTrades
            .map(
              (t) => `<div class="brief-item ${String(t.action || "").toLowerCase()}">
                <b>${actionBadge(t.action)} ${esc(t.symbol)} <span class="mini">${t.target_pct ? `target ${fmtPct(t.target_pct)}` : ""}</span></b>
                <span class="mini">${esc(short(t.reason || "No rationale recorded.", 120))}</span>
              </div>`,
            )
            .join("") || `<div class="empty">No model moves recorded.</div>`
        }
      </div>
    </article>
  `;
}

function renderHero() {
  const snap = state.snapshot;
  const selected = snap.selected_run_summary || {};
  const topPositions = [...snap.positions].sort((a, b) => Number(b.target_pct || 0) - Number(a.target_pct || 0)).slice(0, 3);
  const reasons = topPositions.map((p, idx) => {
    const metrics = [
      ["Allocation", fmtPct(p.target_pct)],
      ["EV 12m", fmtPct(p.ev_12m_pct, true)],
      ["Live value", fmtMoney(p.live_value)],
      ["Sector", p.sector || "Unknown"],
    ];
    return `
      <article class="reason-card">
        <div class="reason-index">${idx + 1}</div>
        <div class="reason-copy">
          <h4>${esc(p.symbol)} ${p.company_name ? `· ${esc(short(p.company_name, 42))}` : ""}</h4>
          <p>${esc(short(p.rationale || "No thesis recorded.", 320))}</p>
          <div class="reason-quote">${esc(p.exit_trigger || p.entry_note || "Model rationale carried forward from the selected run.")}</div>
        </div>
        <div class="reason-metrics">
          <div class="metric-list">
            ${metrics.map(([k, v]) => `<div class="metric-row"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join("")}
          </div>
        </div>
      </article>`;
  }).join("");

  $("heroPanel").innerHTML = `
    <div class="hero-headline">
      <div class="hero-tags">
        <span class="tag buy">Selected run</span>
        <span class="tag good">${selected.buys || 0} buys · ${selected.sells || 0} sells</span>
        <span class="tag info">${selected.positions_selected || snap.positions.length} positions</span>
      </div>
      <div class="hero-title-row">
        <div>
          <h3>Run ${esc(selected.run_id || snap.selected_run || "Latest")}</h3>
          <div class="hero-meta">${esc(selected.date || "")} · ${esc(selected.started_at || "")}</div>
        </div>
      </div>
      <div class="hero-thesis">
        <p>${esc(snap.stage5?.rebalance_notes || "This run collects the latest portfolio construction, rebalance instructions, scenario models, and validation signals into a single operator view.")}</p>
      </div>
      <div class="summary-banner">Expected value ${fmtPct(snap.metrics.weighted_ev_pct, true)} · portfolio target capital ${fmtMoney(snap.metrics.portfolio_value)} · live quote stamp ${esc(snap.metrics.quote_time || "offline")}</div>
      <div class="hero-grid">
        <div class="reason-stack">${reasons || `<div class="empty">No holdings available.</div>`}</div>
        <div class="reason-stack">
          <div class="panel" style="padding:0;border:none;background:transparent">
            <div class="section-kicker">Latest actions</div>
            ${((snap.selected_trades || []).slice(0, 5)).map((t) => `
              <article class="card ${String(t.action || "").toLowerCase()}">
                <div class="card-head">
                  <div>
                    <h4>${esc(t.symbol || "")}</h4>
                    <div class="meta">${esc(t.action || "")} · ${t.target_pct ? fmtPct(t.target_pct) : "n/a"}</div>
                  </div>
                  ${actionBadge(t.action || "HOLD")}
                </div>
                <p>${esc(short(t.reason || "No rationale recorded.", 180))}</p>
              </article>`).join("")}
          </div>
        </div>
      </div>
    </div>
  `;

  $("heroRail").innerHTML = `
    <div class="rail-grid">
      <div class="rail-card feature">
        <div class="rail-kicker">Total tokens</div>
        <strong>${fmtInt(selected.token_total || 0)}</strong>
        <p>${fmtInt(selected.input_tokens || 0)} in / ${fmtInt(selected.output_tokens || 0)} out</p>
      </div>
      <div class="rail-card">
        <div class="rail-kicker">Run time</div>
        <strong>${esc(selected.duration_human || "n/a")}</strong>
        <p>${Math.round(Number(selected.duration_seconds || 0))} seconds</p>
      </div>
      <div class="rail-card">
        <div class="rail-kicker">Portfolio EV</div>
        <strong>${fmtPct(selected.portfolio_ev_pct || snap.metrics.weighted_ev_pct, true)}</strong>
        <p>${selected.positive_ev_count || 0} names screened positive</p>
      </div>
      <div class="rail-card">
        <div class="rail-kicker">Validation</div>
        <strong class="${(selected.warnings || 0) > 0 ? "warn" : "up"}">${selected.warnings || 0}</strong>
        <p>warning items detected</p>
      </div>
    </div>
  `;
}

function renderFindings() {
  const selected = state.snapshot.selected_run_summary || {};
  const warnings = selected.warning_items || [];
  const stage5Trades = state.snapshot.selected_trades || [];
  const findingCards = [];
  if (warnings.length) {
    findingCards.push({
      kind: "sell",
      title: "Validation findings",
      meta: `${warnings.length} surfaced`,
      body: warnings.slice(0, 3).join(" | "),
    });
  }
  if (stage5Trades.length) {
    const buys = stage5Trades.filter((t) => String(t.action).toUpperCase() === "BUY").slice(0, 2).map((t) => t.symbol).join(", ");
    const sells = stage5Trades.filter((t) => String(t.action).toUpperCase() === "SELL").slice(0, 2).map((t) => t.symbol).join(", ");
    findingCards.push({
      kind: "buy",
      title: "Actions taken",
      meta: `${selected.trades || stage5Trades.length} trade instructions`,
      body: `Buys: ${buys || "none"}${sells ? ` · Sells: ${sells}` : ""}`,
    });
  }
  findingCards.push({
    kind: "hold",
    title: "Selected run status",
    meta: `${selected.positions_selected || state.snapshot.positions.length} positions`,
    body: `${selected.positive_ev_count || 0} positive-EV candidates survived modeling and ${selected.holds || 0} holdings were carried or re-rated in the selected cycle.`,
  });
  $("findingsGrid").innerHTML = findingCards.map((card) => `
    <article class="card ${card.kind}">
      <div class="card-head">
        <div>
          <h4>${esc(card.title)}</h4>
          <div class="meta">${esc(card.meta)}</div>
        </div>
      </div>
      <p>${esc(card.body)}</p>
    </article>`).join("");
}

function renderStageBurn() {
  const summary = state.snapshot.run_summary || {};
  const stages = ["stage1", "stage2", "stage3", "stage4", "stage5"].filter((key) => summary[key]);
  $("stageBurn").innerHTML = `
    <div class="burn-grid">
      ${stages.map((key) => {
        const stage = summary[key] || {};
        const tokens = stage.token_usage || {};
        const details = [];
        if (stage.top50_count) details.push(`${stage.top50_count} passed`);
        if (stage.stocks_researched) details.push(`${stage.stocks_researched} researched`);
        if (stage.modeled_count) details.push(`${stage.modeled_count} modeled`);
        if (stage.positions_selected) details.push(`${stage.positions_selected} selected`);
        if (stage.trades) details.push(`${stage.trades} trades`);
        if (stage.positive_ev_count) details.push(`${stage.positive_ev_count} positive EV`);
        return `
          <article class="burn-card">
            <h4>${esc(key.toUpperCase())}</h4>
            <p>${esc(details.join(" · ") || "Completed")}</p>
            <p>${fmtInt(tokens.total_tokens || 0)} tokens</p>
          </article>`;
      }).join("")}
    </div>
    <p class="run-note">Total token burn for this run: ${fmtInt((summary.total_tokens || {}).input || 0)} input + ${fmtInt((summary.total_tokens || {}).output || 0)} output. Duration: ${esc(summary.duration_human || "n/a")}.</p>
  `;
}

function renderKpis() {
  const m = state.snapshot.metrics;
  const kpis = [
    ["Target Capital", fmtMoney(m.portfolio_value), ""],
    ["Live Value", fmtMoney(m.live_value), fmtPct(m.live_return_pct, true)],
    ["Weighted EV", fmtPct(m.weighted_ev_pct, true), "12m model"],
    ["Positions", m.positions, `max ${state.snapshot.rules.max_positions}`],
    ["Urgent Alerts", m.urgent_alerts, m.urgent_alerts ? "attention" : "clear"],
    ["Trades Logged", state.snapshot.trades.length, "all runs"],
  ];
  $("kpiGrid").innerHTML = kpis
    .map(
      ([label, value, sub]) => `
      <div class="kpi">
        <span>${esc(label)}</span>
        <strong>${esc(value)}</strong>
        <small>${esc(sub)}</small>
      </div>`,
    )
    .join("");
}

function renderCharts() {
  if (!state.snapshot) return;
  renderPortfolioChart();
  renderSectorChart();
  renderPriceChart();
}

function renderPortfolioChart() {
  const hist = state.history;
  if (!hist?.dates?.length) {
    $("portfolioChart").innerHTML = `<div class="loading">Price history unavailable.</div>`;
    return;
  }
  const weights = Object.fromEntries(state.snapshot.positions.map((p) => [p.symbol, Number(p.target_pct || 0) / 100]));
  const positionSeries = hist.series.filter((s) => weights[s.name]);
  const portfolio = hist.dates.map((_, idx) =>
    positionSeries.reduce((sum, s) => sum + (Number(s.values[idx] || 0) * weights[s.name]), 0),
  );
  const traces = [{ x: hist.dates, y: portfolio, type: "scatter", mode: "lines", name: "Portfolio", line: { color: "#23c483", width: 3 } }];
  const bench = hist.series.find((s) => s.name === state.snapshot.rules.benchmark);
    if (bench) traces.push({ x: hist.dates, y: bench.values, type: "scatter", mode: "lines", name: BENCHMARK_LABEL, line: { color: "#58a6ff", dash: "dot", width: 2 } });
  Plotly.react("portfolioChart", traces, plotLayout({ yaxis: { title: "Indexed to 100", gridcolor: "#1d2834" } }), plotConfig);
}

function renderSectorChart() {
  const sectors = new Map();
  for (const p of state.snapshot.positions) {
    sectors.set(p.sector || "Unknown", (sectors.get(p.sector || "Unknown") || 0) + Number(p.target_pct || 0));
  }
  Plotly.react(
    "sectorChart",
    [
      {
        labels: [...sectors.keys()],
        values: [...sectors.values()],
        type: "pie",
        hole: 0.58,
        textinfo: "label+percent",
        marker: { line: { color: "#080b0f", width: 2 } },
      },
    ],
    plotLayout({ showlegend: false, margin: { l: 10, r: 10, t: 10, b: 10 } }),
    plotConfig,
  );
}

function renderPriceChart() {
  const selected = [...$("chartSymbols").selectedOptions].map((o) => o.value);
  const hist = state.history;
  if (!hist?.dates?.length) {
    $("priceChart").innerHTML = `<div class="loading">Price history unavailable.</div>`;
    return;
  }
  const traces = hist.series
    .filter((s) => selected.includes(s.name) || s.name === state.snapshot.rules.benchmark)
    .map((s) => ({
      x: hist.dates,
      y: s.values,
      type: "scatter",
      mode: "lines",
      name: s.name === state.snapshot.rules.benchmark ? BENCHMARK_LABEL : s.name,
      line: { width: s.name === state.snapshot.rules.benchmark ? 2 : 2.5, dash: s.name === state.snapshot.rules.benchmark ? "dot" : "solid" },
    }));
  Plotly.react("priceChart", traces, plotLayout({ yaxis: { title: "Indexed to 100", gridcolor: "#1d2834" } }), plotConfig);
}

function renderOverviewTables() {
  const top = [...state.snapshot.positions].sort((a, b) => Number(b.ev_12m_pct || 0) - Number(a.ev_12m_pct || 0)).slice(0, 6);
  $("topEvTable").innerHTML = table(
    [
      { key: "symbol", label: "Symbol" },
      { key: "target", label: "Target", num: true },
      { key: "ev", label: "EV", num: true },
    ],
    top.map((p) => ({ symbol: p.symbol, target: fmtPct(p.target_pct), ev: fmtPct(p.ev_12m_pct, true) })),
  );

  const drift = [...state.snapshot.positions].sort((a, b) => Math.abs(Number(b.drift_pct || 0)) - Math.abs(Number(a.drift_pct || 0))).slice(0, 6);
  $("driftTable").innerHTML = table(
    [
      { key: "symbol", label: "Symbol" },
      { key: "live", label: "Live", num: true },
      { key: "drift", label: "Drift", num: true },
    ],
    drift.map((p) => ({ symbol: p.symbol, live: fmtPct(p.live_pct), drift: fmtPct(p.drift_pct, true) })),
  );

  const recent = [...(state.snapshot.selected_trades || [])].slice(0, 6);
  $("recentDecisions").innerHTML = table(
    [
      { key: "action", label: "Action" },
      { key: "symbol", label: "Symbol" },
      { key: "target", label: "Target", num: true },
    ],
    recent.map((t) => ({ action: t.action, symbol: t.symbol, target: t.target_pct ? fmtPct(t.target_pct) : "" })),
  );
}

function renderRunLibrary() {
  const rows = (state.snapshot.run_catalog || []).map((run) => ({
    date: run.date || "",
    run: run.run_id,
    duration: run.duration_human || "",
    tokens: fmtInt(run.token_total || 0),
    stages: (run.stages || []).join(" / "),
    ev: run.portfolio_ev_pct ? fmtPct(run.portfolio_ev_pct, true) : "",
    trades: `${run.trades || 0} (${run.buys || 0}B/${run.sells || 0}S/${run.holds || 0}H)`,
    findings: run.warnings ? `${run.warnings} warnings` : "clean",
    symbols: (run.top_symbols || []).join(", "),
  }));
  $("runLibrary").innerHTML = table(
    [
      { key: "date", label: "Date" },
      { key: "run", label: "Run ID" },
      { key: "duration", label: "Runtime" },
      { key: "tokens", label: "Tokens", num: true },
      { key: "stages", label: "Stages" },
      { key: "ev", label: "EV", num: true },
      { key: "trades", label: "Actions" },
      { key: "findings", label: "Findings" },
      { key: "symbols", label: "Main symbols" },
    ],
    rows,
  );
}

function renderMarket() {
  const query = $("marketSearch").value.toLowerCase();
  const rows = state.snapshot.positions
    .filter((p) => JSON.stringify(p).toLowerCase().includes(query))
    .map((p) => ({
      symbol: `<b>${esc(p.symbol)}</b>`,
      company: short(p.company_name, 34),
      sector: p.sector || "",
      price: fmtPrice(p.live_price),
      target: fmtMoney(p.target_value),
      live: fmtMoney(p.live_value),
      weight: fmtPct(p.live_pct),
      drift: `<span class="${Number(p.drift_pct || 0) >= 0 ? "up" : "down"}">${fmtPct(p.drift_pct, true)}</span>`,
      quote: short(p.quote_time, 20),
    }));
  $("marketTable").innerHTML = table(
    [
      { key: "symbol", label: "Symbol" },
      { key: "company", label: "Company" },
      { key: "sector", label: "Sector" },
      { key: "price", label: "Price", num: true },
      { key: "target", label: "Target", num: true },
      { key: "live", label: "Live", num: true },
      { key: "weight", label: "Weight", num: true },
      { key: "drift", label: "Drift", num: true },
      { key: "quote", label: "Quote" },
    ],
    rows,
    { raw: ["symbol", "drift"] },
  );

  const select = $("chartSymbols");
  if (!select.options.length) {
    select.innerHTML = state.snapshot.positions
      .map((p, i) => `<option value="${esc(p.symbol)}"${i < 5 ? " selected" : ""}>${esc(p.symbol)}</option>`)
      .join("");
  }
}

function renderFilters() {
  const sectors = ["All", ...new Set(state.snapshot.positions.map((p) => p.sector || "Unknown").sort())];
  const convictions = ["All", ...new Set(state.snapshot.positions.map((p) => p.conviction || "Medium").sort())];
  fillSelect("sectorFilter", sectors);
  fillSelect("convictionFilter", convictions);
  fillSelect("decisionAction", ["All", ...new Set(state.snapshot.trades.map((t) => t.action).sort())]);
  fillSelect("decisionRun", ["All", ...new Set(state.snapshot.trades.map((t) => t.run_id).filter(Boolean).sort().reverse())]);
  fillSelect("scenarioSelect", state.snapshot.scenario_models.map((m) => m.symbol).filter(Boolean));
}

function fillSelect(id, values) {
  const el = $(id);
  const current = el.value;
  el.innerHTML = values.map((v) => `<option value="${esc(v)}"${v === current ? " selected" : ""}>${esc(v)}</option>`).join("");
}

function renderHoldings() {
  const sector = $("sectorFilter").value || "All";
  const conviction = $("convictionFilter").value || "All";
  const q = $("holdingSearch").value.toLowerCase();
  const rows = state.snapshot.positions.filter((p) => {
    if (sector !== "All" && (p.sector || "Unknown") !== sector) return false;
    if (conviction !== "All" && (p.conviction || "Medium") !== conviction) return false;
    return JSON.stringify(p).toLowerCase().includes(q);
  });
  $("holdingsList").innerHTML = rows
    .sort((a, b) => Number(b.target_pct || 0) - Number(a.target_pct || 0))
    .map(
      (p) => `
      <article class="card ${String(p.action || "hold").toLowerCase()}">
        <div class="card-head">
          <div>
            <h4>${esc(p.symbol)} · ${esc(short(p.company_name, 58))}</h4>
            <div class="meta">${esc(p.sector || "Unknown")} · ${esc(p.conviction || "Medium")} conviction</div>
          </div>
          ${actionBadge(p.action || "HOLD")}
        </div>
        <div class="split">
          <div><span>Target</span><strong>${fmtPct(p.target_pct)}</strong></div>
          <div><span>Live weight</span><strong>${fmtPct(p.live_pct)}</strong></div>
          <div><span>Live price</span><strong>${fmtPrice(p.live_price)}</strong></div>
          <div><span>EV 12m</span><strong>${fmtPct(p.ev_12m_pct, true)}</strong></div>
        </div>
        <p><b>Thesis:</b> ${esc(p.rationale || "No thesis recorded.")}</p>
        <p><b>Entry:</b> ${esc(p.entry_note || "No entry note recorded.")}</p>
        <p><b>Exit trigger:</b> ${esc(p.exit_trigger || "No exit trigger recorded.")}</p>
      </article>`,
    )
    .join("") || `<div class="empty">No holdings match the filters.</div>`;
}

function renderDecisions() {
  const action = $("decisionAction").value || "All";
  const run = $("decisionRun").value || "All";
  const q = $("decisionSearch").value.toLowerCase();
  const rows = state.snapshot.trades.filter((t) => {
    if (action !== "All" && t.action !== action) return false;
    if (run !== "All" && t.run_id !== run) return false;
    return JSON.stringify(t).toLowerCase().includes(q);
  });
  $("decisionTable").innerHTML = table(
    [
      { key: "date", label: "Date" },
      { key: "run", label: "Run" },
      { key: "action", label: "Action" },
      { key: "symbol", label: "Symbol" },
      { key: "current", label: "Current", num: true },
      { key: "target", label: "Target", num: true },
      { key: "ev", label: "EV", num: true },
      { key: "reason", label: "Rationale" },
    ],
    rows.map((t) => ({
      date: t.date || "",
      run: t.run_id || "",
      action: t.action,
      symbol: t.symbol,
      current: t.current_pct ? fmtPct(t.current_pct) : "",
      target: t.target_pct ? fmtPct(t.target_pct) : "",
      ev: fmtPct(t.ev_12m_pct, true),
      reason: short(t.reason, 90),
    })),
  );
  $("decisionCards").innerHTML = rows
    .slice()
    .reverse()
    .slice(0, 80)
    .map(
      (t) => `
      <article class="card ${String(t.action || "").toLowerCase()}">
        <div class="card-head">
          <div>
            <h4>${esc(t.symbol)} · ${esc(short(t.company_name, 54))}</h4>
            <div class="meta">${esc(t.date || "")} · ${esc(t.run_id || "")} · target ${t.target_pct ? fmtPct(t.target_pct) : "n/a"} · EV ${fmtPct(t.ev_12m_pct, true)}</div>
          </div>
          ${actionBadge(t.action)}
        </div>
        <p>${esc(t.reason || "No rationale recorded.")}</p>
        <p class="meta"><b>Counterargument:</b> ${esc(t.counterargument || "Not recorded.")}</p>
      </article>`,
    )
    .join("") || `<div class="empty">No decisions match the filters.</div>`;
}

function renderRebalance() {
  const stage5 = state.snapshot.stage5 || {};
  const summary = stage5.rebalance_summary || {};
  const items = [
    ["Buys", summary.buys || 0],
    ["Sells", summary.sells || 0],
    ["Holds", summary.holds || 0],
    ["Turnover", fmtPct(summary.turnover_pct || 0)],
    ["EV Lift", fmtPct(Math.abs(summary.ev_improvement || 0) <= 1.5 ? (summary.ev_improvement || 0) * 100 : summary.ev_improvement || 0, true)],
  ];
  $("rebalanceKpis").innerHTML = items.map(([k, v]) => `<div class="mini-kpi"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join("");
  $("rebalanceNotes").textContent = stage5.rebalance_notes || "No rebalance notes recorded.";
  const trades = state.snapshot.selected_trades || [];
  $("rebalanceTable").innerHTML = table(
    [
      { key: "action", label: "Action" },
      { key: "symbol", label: "Symbol" },
      { key: "company", label: "Company" },
      { key: "sector", label: "Sector" },
      { key: "target", label: "Target", num: true },
      { key: "ev", label: "EV", num: true },
      { key: "reason", label: "Reason" },
      { key: "counter", label: "Counterargument" },
    ],
    trades.map((t) => ({
      action: t.action,
      symbol: t.symbol,
      company: short(t.company_name, 34),
      sector: t.sector || "",
      target: t.target_pct ? fmtPct(t.target_pct) : "",
      ev: fmtPct(t.ev_12m_pct, true),
      reason: short(t.reason, 120),
      counter: short(t.counterargument, 90),
    })),
  );
  const counts = ["BUY", "SELL", "HOLD"].map((a) => trades.filter((t) => String(t.action).toUpperCase() === a).length);
  Plotly.react(
    "rebalanceChart",
    [{ x: ["Buy", "Sell", "Hold"], y: counts, type: "bar", marker: { color: ["#23c483", "#ef6262", "#e3a528"] }, text: counts, textposition: "outside" }],
    plotLayout({ showlegend: false, margin: { l: 40, r: 10, t: 10, b: 36 } }),
    plotConfig,
  );
}

function renderScenarios() {
  const models = [...state.snapshot.scenario_models].sort((a, b) => Number(b.ev_12m_pct || 0) - Number(a.ev_12m_pct || 0));
  $("scenarioTable").innerHTML = table(
    [
      { key: "symbol", label: "Symbol" },
      { key: "company", label: "Company" },
      { key: "sector", label: "Sector" },
      { key: "rec", label: "Recommendation" },
      { key: "ev", label: "EV", num: true },
      { key: "cases", label: "Cases", num: true },
    ],
    models.map((m) => ({ symbol: m.symbol, company: short(m.company_name, 34), sector: m.sector || "", rec: m.recommendation || "", ev: fmtPct(m.ev_12m_pct, true), cases: m.cases?.length || 0 })),
  );
  const selected = $("scenarioSelect").value || models[0]?.symbol;
  const model = models.find((m) => m.symbol === selected) || models[0];
  if (!model) {
    $("scenarioTitle").textContent = "Scenario Detail";
    $("scenarioCases").innerHTML = `<div class="empty">No scenario models found.</div>`;
    $("debateBox").innerHTML = "";
    return;
  }
  $("scenarioTitle").textContent = `${model.symbol} · ${model.company_name || ""}`;
  const cases = model.cases || [];
  Plotly.react(
    "scenarioChart",
    [{ x: cases.map((c) => c.case), y: cases.map((c) => c.return_pct || 0), type: "bar", marker: { color: cases.map((c) => (Number(c.return_pct || 0) >= 0 ? "#23c483" : "#ef6262")) } }],
    plotLayout({ showlegend: false, yaxis: { title: "Return %", gridcolor: "#1d2834" }, margin: { l: 44, r: 10, t: 10, b: 36 } }),
    plotConfig,
  );
  $("scenarioCases").innerHTML = table(
    [
      { key: "case", label: "Case" },
      { key: "prob", label: "Prob.", num: true },
      { key: "target", label: "Target", num: true },
      { key: "return", label: "Return", num: true },
      { key: "scenario", label: "Scenario" },
    ],
    cases.map((c) => ({ case: c.case, prob: fmtPct(c.probability), target: c.target ? fmtPrice(c.target) : "", return: fmtPct(c.return_pct, true), scenario: c.scenario || "" })),
  );
  const bull = (model.debate_log?.bull_case || []).join("\n\n");
  const bear = (model.debate_log?.bear_case || []).join("\n\n");
  $("debateBox").innerHTML = `
    <div class="card buy"><h4>Bull case</h4><p>${esc(short(bull, 1800) || "No bull case recorded.")}</p></div>
    <div class="card sell"><h4>Bear case</h4><p>${esc(short(bear, 1800) || "No bear case recorded.")}</p></div>
    ${model.debate_log?.resolution ? `<div class="card hold"><h4>Resolution</h4><p>${esc(model.debate_log.resolution)}</p></div>` : ""}
  `;
}

function renderMonitor() {
  const alerts = state.snapshot.thesis_alerts || {};
  const dates = Object.keys(alerts).sort().reverse().slice(0, 10);
  $("monitorList").innerHTML =
    dates
      .map((date) => {
        const all = alerts[date].all_results || [];
        return `<article class="card">
          <h4>${esc(date)} · ${all.length} checks · ${(alerts[date].urgent || []).length} urgent</h4>
          ${all
            .map((r) => `<p><b>${esc(r.symbol || "")}</b> <span class="${r.alert_level === "URGENT" ? "down" : r.alert_level === "WATCH" ? "warn" : "up"}">${esc(r.alert_level || "NONE")}</span> → ${esc(r.action_recommended || "HOLD")}<br><span class="meta">${esc(r.reason || r.news_summary || "No significant thesis change.")}</span></p>`)
            .join("")}
        </article>`;
      })
      .join("") || `<div class="empty">No thesis monitor data yet.</div>`;

  const db = state.snapshot.db || {};
  const countRows = Object.entries(db)
    .filter(([, v]) => Number.isFinite(v))
    .map(([k, v]) => ({ table: k, rows: v }));
  $("sourceHealth").innerHTML = `
    <div class="card">
      <h4>Latest run</h4>
      <p>${esc(JSON.stringify(db.latest_run || state.snapshot.run_summary || {}, null, 2))}</p>
    </div>
    ${table([{ key: "table", label: "Table" }, { key: "rows", label: "Rows", num: true }], countRows)}
  `;
}

function bindEvents() {
  $("refreshBtn").addEventListener("click", loadSnapshot);
  $("runSelect").addEventListener("change", loadSnapshot);
  $("liveToggle").addEventListener("change", loadSnapshot);
  $("periodSelect").addEventListener("change", loadHistory);
  $("marketSearch").addEventListener("input", renderMarket);
  $("chartSymbols").addEventListener("change", renderPriceChart);
  $("sectorFilter").addEventListener("change", renderHoldings);
  $("convictionFilter").addEventListener("change", renderHoldings);
  $("holdingSearch").addEventListener("input", renderHoldings);
  $("decisionAction").addEventListener("change", renderDecisions);
  $("decisionRun").addEventListener("change", renderDecisions);
  $("decisionSearch").addEventListener("input", renderDecisions);
  $("scenarioSelect").addEventListener("change", renderScenarios);
  $("nav").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-view]");
    if (!button) return;
    state.activeView = button.dataset.view;
    document.querySelectorAll(".nav button").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${state.activeView}`));
    setTimeout(() => window.dispatchEvent(new Event("resize")), 20);
  });
}

async function boot() {
  bindEvents();
  updateClock();
  setInterval(updateClock, 1000);
  try {
    await loadSnapshot();
    setInterval(loadSnapshot, 5 * 60 * 1000);
  } catch (error) {
    document.querySelector(".main").innerHTML = `<div class="panel"><h3>Dashboard failed to load</h3><p>${esc(error.message)}</p></div>`;
  }
}

boot();
