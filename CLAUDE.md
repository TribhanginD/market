# CLAUDE.md

## Offlimit Files
- `.env` — contains live API keys. Hard-blocked via `.claude/settings.json` `permissions.deny`. Do not Read, cat, grep, or otherwise output contents. Runtime access via `os.environ` is fine.

## High-Leverage Moves (ranked for returns)

Status: M1–M6 shipped. M1–M3 (see git diff for `data/broker_feeds.py`, debate slicing, `pipeline/backtest.py`). M4–M6 below.

1. [DONE] **Broker upgrade/downgrade + earnings revision feed** → Stage 2 input. Single biggest alpha source. Files: `data/broker_feeds.py`, wired in `pipeline/stage2_adversarial.py`.
2. [DONE] **Per-agent unique data sources** — enforce in debate engine that each persona only sees its slice; prevents echo. File: `agents/debate_engine.py` (`_slice_data_context_for`).
3. [DONE] **Backtest harness** — replay historical pipeline on past N quarters, score EV calibration. Without this you don't know if debate adds alpha. File: `pipeline/backtest.py`, CLI: `--mode backtest --horizon-days N`.
4. [DONE] **Regime-aware prompts** — macro agent emits `regime ∈ {risk_on, risk_off, rotation}`; downstream agents condition prompts on it. Touch: `agents/personas.py` (or wherever MACRO_SYSTEM_PROMPT lives) + `agents/debate_engine.py` (pass regime to other agents via ctx).
5. [DONE] **Sentiment quant layer** — daily NSE announcement scrape + FinBERT (or India-tuned) sentiment score per stock, fed into Stage 1 composite score. Free model: HuggingFace `ProsusAI/finbert` or `yiyanghkust/finbert-tone`. Touch: `data/sentiment.py` (new), `pipeline/stage1_screening.py`.
6. [DONE] **Auto-rerun trigger** — thesis-break event in `pipeline/thesis_monitor.py` enqueues partial pipeline rerun (just affected symbol → Stage 2-4), not whole-universe rerun. Touch: monitor + orchestrator new partial-mode entrypoint.

## True Goal

Recreate `@claudeportfolio` method for Indian markets.

Not single prompt. Not vibe check. System = multi-agent orchestration pipeline built on Claude-style agent chaining.

Original public pattern:

- full universe, no human stock picking
- 5 chained stages
- adversarial bull vs bear research
- probability-weighted scenario models
- optimizer picks exact portfolio
- full rerun every cycle
- thesis monitoring between cycles
- every trade must explain thesis, catalysts, risks, expected return

India adaptation:

- universe = Nifty 500 or similar liquid India universe
- benchmark = Nifty 50 or Nifty 500
- inputs = NSE/BSE filings, India news, India macro, broker research, Screener/Moneycontrol-style fundamentals

## North Star

Agents must disagree usefully.

Bad:

- agents repeat same thesis
- bear side weak strawman
- manager rubber-stamps consensus
- scenario model copies bull case

Good:

- bull agents hunt upside
- bear agents hunt thesis breaks
- manager forces unresolved questions into output
- final pick beats alternatives, not merely sounds good

## First Read

1. `README.md`
2. `run.py`
3. `config.py`
4. `pipeline/orchestrator.py`
5. files in `agents/` and `pipeline/` for touched stage

## True Entrypoints

- `run.py` = main CLI
- `config.py` = single source for knobs, paths, models, limits
- `pipeline/orchestrator.py` = stage chain
- `dashboard/server.py` = live dashboard server

Do not add new entrypoint unless old path blocked.

## Exact Pipeline Shape

### Stage 1: Screening

- score full universe on fundamentals, momentum, macro fit
- no human pre-selection
- output ranked list
- top names advance

### Stage 2: Adversarial Research

- bull side argues why buy
- bear side argues why avoid / sell / size down
- recent info matters most
- arguments must cite concrete facts, not tone
- output must preserve disagreement, not flatten too early

### Stage 3: Scenario Modeling

- build bull / base / bear
- assign probabilities
- estimate 1 / 3 / 6 / 12 month targets when possible
- compute probability-weighted EV
- this stage is key filter against blind agreement

### Stage 4: Portfolio Construction

- choose exact positions and weights
- enforce hard constraints
- positive EV only
- diversification real, not cosmetic
- portfolio must beat benchmark in model

### Stage 5: Rebalance

- rerun full pipeline
- compare fresh outputs vs current holdings
- trade only if better opportunity or thesis break
- rationale must explain why switch beats hold

### Between Runs

- daily thesis monitoring
- no dumb stop-loss by price alone
- exit when thesis or facts break

## Repo Shape

- `agents/` = role agents, debate, synthesis, memory
- `data/` = fetch, universe, official sources, reports, cache
- `pipeline/` = stages 1-5, ETL, monitor, validators
- `persistence/` = SQLite schema + writes
- `dashboard/` = HTTP server + static UI
- `storage/` = generated state, payloads, runs, DB, PDFs

## Hard Rules

- Keep `config.py` central. New tunable? put there first.
- Keep Stage 4 and Stage 5 deterministic where possible.
- Preserve artifact contract. Runs write under `storage/pipeline_runs/<run_id>/`.
- Preserve portfolio contract. `storage/portfolio.json`, `storage/trades_log.json`, `storage/portfolio.sqlite3` stay coherent.
- Do not commit generated junk. Respect `.gitignore` and `.codexignore`.
- Do not delete user data in `storage/` unless task says so.
- Do not add fake agent count theater. More agents only if they create distinct views.

## Debate Rules

- Agent personas need different incentives, not cosmetic names.
- Bull agent must attack bear claim. Bear agent must attack bull claim.
- Manager must ask: what is missing, what is unpriced, what breaks thesis.
- Synthesis must preserve strongest pro and strongest con.
- If all agents agree too fast, treat as failure mode.
- Consensus earned. Never default.

## Output Rules

Every candidate or trade should aim to carry:

- thesis
- why now
- catalysts
- risks
- probability-weighted return
- alternative considered
- reason chosen over alternatives

Do not output generic “strong company, good long-term story” junk.

## High-Risk Areas

- `config.py`: small env change can alter whole system behavior
- `agents/debate_engine.py`: false consensus, aggregation bugs, agent overwrite bugs
- `pipeline/stage2_adversarial.py`: research breadth, debate packet shape
- `pipeline/stage3_scenarios.py`: EV math, normalization, scenario shape
- `pipeline/stage4_construction.py`: position caps, sector caps, min/max positions
- `pipeline/stage5_rebalance.py`: buy/sell diff logic, first-run behavior
- `data/official_sources.py`: source normalization, payload persistence
- `persistence/db.py`: schema drift breaks dashboard + pipeline history

Touch these carefully. Verify with tests or smoke run.

### codex review and suggestions

Codex review verdict: strong architecture, but still a powerful prototype rather than production-ready autonomous operation.

Highest-priority fixes:

1. Protect portfolio persistence from bad Stage 5 output.
   - If Stage 5 parse/schema validation fails, do not persist trades, portfolio JSON, or current DB positions.
   - Empty `trades` from an error path must never wipe the current portfolio.

2. Make validation failures fatal before persistence.
   - Inter-stage validation should fail the run for allocation, probability, EV traceability, or duplicate-debate issues.
   - Warnings are fine for dashboard display, but not for accepting a completed autonomous run.

3. Harden DB run lifecycle.
   - Do not silently ignore DB write failures for run start/finalization.
   - Add stale-running-run recovery or mark interrupted runs as failed/stale.
   - Keep `runs`, `positions`, `current_positions`, `trades`, and JSON artifacts coherent.

4. Clean provider/config drift.
   - Remove stale Lightning provider references.
   - Align ETL default model with the chosen ETL provider policy.
   - Keep provider routing documented in one place and covered by tests.

5. Reconcile fallback behavior with the no-fallback production rule.
   - Critical live data and source failures should fail closed during production runs.
   - Any demo/test fallback must be explicit, logged, and gated by mode/config.

6. Improve reproducibility and repo hygiene.
   - Pin dependency versions or add a lockfile.
   - Keep generated `storage/` artifacts, browser profiles, backtests, and caches clearly separated from source.
   - Add ignore rules for any generated runtime state not meant for source control.

Recommended implementation order:
1. Stage 5 schema validation and no-persist-on-invalid-output.
2. Fatal pipeline validation before Stage 5/persistence.
3. DB lifecycle recovery and stale run cleanup.
4. Provider cleanup and ETL model default alignment.
5. Fallback-mode gating.
6. Dependency and generated-artifact hygiene.

## Before Big Changes

- grep call sites
- inspect tests for module
- inspect JSON output shape
- inspect dashboard/API readers if data shape changes
- ask: does this improve disagreement quality or only add complexity

## Preferred Changes

- small patch over broad rewrite
- reuse current JSON fields when possible
- add helper fn before duplicating logic
- fail loud on missing critical data
- silent fallback only when intentional and logged

## Data + Storage Rules

- treat `storage/` as runtime state, not source code
- sidecars like `.sqlite3-wal`, caches, browser profiles = disposable
- raw payloads and PDFs can be large; avoid scanning whole tree unless needed
- if new generated dir appears, add ignore rule

## Testing

Run narrow tests first.

```bash
pytest -q tests/test_source_registry.py
pytest -q tests/test_official_sources.py
pytest -q tests/test_portfolio_constraints.py
pytest -q tests/test_openai_compat_config.py
pytest -q tests/test_debate_system.py
```

If full suite fails, say if break is preexisting. Do not hide it.

## Common Commands

```bash
python run.py --mode test-data
python run.py --mode sync-sources
python run.py --mode etl
python run.py --mode pipeline --stages 1-3
python run.py --mode status
python run.py --mode dashboard
```

## Good Change Shape

- one behavior, one invariant, one bug class
- tests for changed contract
- no unrelated storage churn
- docs only if operator behavior changed

## Avoid

- deleting live storage casually
- renaming fields without grep + migration thought
- hiding failed fetch/parse paths
- mixing cleanup with logic rewrite unless task says so
- adding dead demos, duplicate dashboards, or fake multi-agent wrappers
- making agents polite yes-men
