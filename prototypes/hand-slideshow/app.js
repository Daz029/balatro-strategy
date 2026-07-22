/* Balatro-style slideshow for dumped hand-evaluation decisions. */

const app = document.querySelector("#app");
const params = new URLSearchParams(window.location.search);
const state = {
  runs: [],
  seed: params.get("seed"),
  frames: [],
  frameIndex: 0,
  showPickOrder: false,
};

const esc = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const formatNumber = (value) =>
  value === null || value === undefined ? "\u2014" : Number(value).toLocaleString();

function suitGlyph(suit) {
  return { Spades: "\u2660", Hearts: "\u2665", Clubs: "\u2663", Diamonds: "\u2666" }[suit] || "?";
}

function rankLabel(rank) {
  return { Ace: "A", King: "K", Queen: "Q", Jack: "J" }[rank] || rank || "?";
}

function isRedSuit(suit) {
  return suit === "Hearts" || suit === "Diamonds";
}

function actionPresentation(frame) {
  const isDiscard = frame.action_type === "Discard";
  return {
    cardLabel: isDiscard ? "DISCARD" : "PLAY",
    heading: isDiscard ? "Discard cards" : "Play hand",
    resultLabel: isDiscard ? "MODEL ACTION" : "SCORING HAND",
    resultValue: isDiscard ? `${frame.selected_indices.length} CARDS` : frame.hand_type || "Unknown",
    resultDetail:
      frame.hand_chips === null || frame.hand_chips === undefined
        ? "No score for a discard"
        : `${formatNumber(frame.hand_chips)} chips \u00d7 ${formatNumber(frame.hand_mult)} mult`,
  };
}

function updateUrl() {
  const frame = state.frames[state.frameIndex];
  const next = new URLSearchParams(window.location.search);
  if (state.seed) next.set("seed", state.seed);
  if (frame) next.set("decision", frame.decision);
  window.history.replaceState({}, "", `${window.location.pathname}?${next}`);
}

function transportHTML(frame) {
  const options = state.runs
    .map(
      (run) =>
        `<option value="${esc(run.seed)}" ${run.seed === state.seed ? "selected" : ""}>${esc(
          run.seed,
        )} \u00b7 ${run.frames} decisions</option>`,
    )
    .join("");
  return `<header class="transport">
    <div class="brand"><span>POLICY</span><strong>HAND REPLAY</strong></div>
    <label class="run-select">RUN<select data-control="seed">${options}</select></label>
    <nav class="step-controls" aria-label="Decision navigation">
      <button data-action="previous" aria-label="Previous decision" ${state.frameIndex === 0 ? "disabled" : ""}>\u25c0</button>
      <button class="step" data-action="next" ${state.frameIndex === state.frames.length - 1 ? "disabled" : ""}>STEP \u25b6</button>
      <label>DECISION <input id="decision-input" type="number" min="1" value="${esc(frame.decision)}" /></label>
      <button data-action="jump">GO</button>
    </nav>
    <div class="frame-count"><strong>${state.frameIndex + 1}</strong> / ${state.frames.length}<small>SPACE TO STEP</small></div>
  </header>`;
}

function statHTML(label, value, className = "", prefix = "") {
  return `<div class="stat ${className}"><small>${label}</small><strong>${prefix}${formatNumber(value)}</strong></div>`;
}

function statusRailHTML(frame) {
  const progress = Number(frame.required_score)
    ? Math.min(100, (Number(frame.current_score) / Number(frame.required_score)) * 100)
    : 0;
  return `<aside class="status-rail">
    ${statHTML("REQUIRED SCORE", frame.required_score, "required")}
    <div class="score-progress"><i style="width:${progress}%"></i></div>
    ${statHTML("CURRENT SCORE", frame.current_score, "current")}
    ${statHTML("HAND SCORE", frame.hand_score, "hand-score")}
    ${statHTML("MONEY", frame.money, "money", "$")}
    <div class="blind-name ${frame.blind.boss ? "boss" : ""}"><small>${frame.blind.boss ? "BOSS BLIND" : "CURRENT BLIND"}</small><strong>${esc(frame.blind.name)}</strong></div>
    <div class="turns-left">
      ${statHTML("HANDS LEFT", frame.hands_left, "hands")}
      ${statHTML("DISCARDS LEFT", frame.discards_left, "discards")}
    </div>
    <div class="ante"><small>ANTE</small><strong>${formatNumber(frame.ante)}</strong><span>ROUND ${formatNumber(frame.round)}</span></div>
  </aside>`;
}

function jokerHTML(joker) {
  const flags = [
    joker.edition,
    joker.eternal && "ETERNAL",
    joker.perishable && "PERISHABLE",
    joker.rental && "RENTAL",
    joker.debuff && "DEBUFFED",
  ]
    .filter(Boolean)
    .join(" \u00b7 ");
  return `<article class="joker ${joker.debuff ? "debuffed" : ""}" title="${esc(joker.key || joker.name)}">
    <span class="joker-star">\u2605</span>
    <strong>${esc(joker.name)}</strong>
    <small>${esc(joker.detail)}</small>
    ${flags ? `<em>${esc(flags)}</em>` : ""}
  </article>`;
}

function jokersHTML(frame) {
  const jokers = frame.jokers.length
    ? frame.jokers.map(jokerHTML).join("")
    : '<div class="empty-jokers"><span>NO JOKERS</span></div>';
  return `<section class="joker-zone">
    <div class="zone-label"><strong>JOKERS</strong><span>${frame.jokers.length} / 8 SHOWN</span></div>
    <div class="joker-rack" style="--joker-count:${Math.max(1, frame.jokers.length)}">${jokers}</div>
  </section>`;
}

function cardHTML(card, actionLabel) {
  const red = isRedSuit(card.suit);
  const details = [card.enhancement, card.edition, card.seal && `${card.seal} seal`]
    .filter(Boolean)
    .join(" \u00b7 ");
  return `<article class="playing-card ${red ? "red" : "black"} ${card.picked ? "picked" : ""} ${card.debuff ? "debuffed" : ""}" title="Original hand slot ${card.slot + 1}">
    ${card.picked ? `<span class="selection-label">${actionLabel}</span>` : ""}
    ${card.picked ? `<span class="pick-order" aria-hidden="${state.showPickOrder ? "false" : "true"}">${card.pick_order}</span>` : ""}
    <div class="corner"><b>${esc(rankLabel(card.rank))}</b><i>${suitGlyph(card.suit)}</i></div>
    <div class="suit">${suitGlyph(card.suit)}</div>
    <div class="card-name"><strong>${esc(card.rank)}</strong><span>${esc(card.suit)}</span></div>
    ${details ? `<small>${esc(details)}</small>` : ""}
  </article>`;
}

function handHTML(frame) {
  const action = actionPresentation(frame);
  const cards = frame.hand.map((card) => cardHTML(card, action.cardLabel)).join("");
  const overflow = frame.hidden_hand_count
    ? `<span>${frame.hidden_hand_count} unselected overflow card${frame.hidden_hand_count === 1 ? "" : "s"} hidden</span>`
    : "";
  return `<section class="hand-zone">
    <div class="decision-heading">
      <div><small>DECISION ${frame.decision}</small><h1>${action.heading}</h1></div>
      <div class="decision-result"><small>${action.resultLabel}</small><strong>${esc(action.resultValue)}</strong><span>${action.resultDetail}</span></div>
    </div>
    <div class="hand-meta"><strong>${frame.hand_count} CARDS IN HAND</strong>${overflow}</div>
    <div class="hand-row" style="--card-count:${Math.max(1, frame.hand.length)}">${cards}</div>
  </section>`;
}

function pickOrderHTML(frame) {
  const picks = [...frame.hand]
    .filter((card) => card.picked)
    .sort((left, right) => left.pick_order - right.pick_order);
  const orderedCards = picks
    .map(
      (card) =>
        `<li><b>${card.pick_order}</b><span class="${isRedSuit(card.suit) ? "red-text" : ""}">${esc(rankLabel(card.rank))}${suitGlyph(card.suit)}</span></li>`,
    )
    .join("");
  return `<div class="pick-order-control ${state.showPickOrder ? "open" : ""}">
    <div class="pick-order-panel" aria-hidden="${state.showPickOrder ? "false" : "true"}">
      <small>RECORDED PICK ORDER</small><ol>${orderedCards}</ol>
    </div>
    <button data-action="toggle-order" aria-pressed="${state.showPickOrder}">${state.showPickOrder ? "HIDE" : "SHOW"} PICK ORDER</button>
  </div>`;
}

function render() {
  const frame = state.frames[state.frameIndex];
  if (!frame) return;
  app.innerHTML = `<div class="viewer ${state.showPickOrder ? "show-pick-order" : ""}">
    ${transportHTML(frame)}
    <div class="felt">${statusRailHTML(frame)}${jokersHTML(frame)}${handHTML(frame)}${pickOrderHTML(frame)}</div>
  </div>`;
  updateUrl();
}

function moveFrame(delta) {
  state.frameIndex = Math.max(0, Math.min(state.frames.length - 1, state.frameIndex + delta));
  render();
}

function jumpToDecision(rawDecision) {
  const decision = Number(rawDecision);
  if (!Number.isFinite(decision)) return;
  let closest = 0;
  for (let index = 1; index < state.frames.length; index += 1) {
    if (
      Math.abs(state.frames[index].decision - decision) <
      Math.abs(state.frames[closest].decision - decision)
    ) {
      closest = index;
    }
  }
  state.frameIndex = closest;
  render();
}

async function loadRun(seed, requestedDecision = null) {
  const response = await fetch(`/api/frames?seed=${encodeURIComponent(seed)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Could not load ${seed}`);
  state.seed = seed;
  state.frames = payload.frames || [];
  state.frameIndex = 0;
  if (requestedDecision !== null) jumpToDecision(requestedDecision);
  else render();
}

app.addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  if (button.dataset.action === "next") moveFrame(1);
  if (button.dataset.action === "previous") moveFrame(-1);
  if (button.dataset.action === "jump") {
    jumpToDecision(document.querySelector("#decision-input")?.value);
  }
  if (button.dataset.action === "toggle-order") {
    state.showPickOrder = !state.showPickOrder;
    render();
  }
});

app.addEventListener("change", (event) => {
  if (event.target.matches('[data-control="seed"]')) loadRun(event.target.value);
});

app.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.matches("#decision-input")) {
    jumpToDecision(event.target.value);
  }
});

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, textarea, [contenteditable]")) return;
  if (event.key === "ArrowRight") moveFrame(1);
  if (event.key === "ArrowLeft") moveFrame(-1);
  if (event.code === "Space") {
    event.preventDefault();
    moveFrame(event.shiftKey ? -1 : 1);
  }
  if (event.key.toLowerCase() === "o") {
    state.showPickOrder = !state.showPickOrder;
    render();
  }
});

async function init() {
  try {
    const response = await fetch("/api/runs");
    state.runs = await response.json();
    const seed = state.runs.some((run) => run.seed === state.seed)
      ? state.seed
      : state.runs[0]?.seed;
    if (!seed) throw new Error("No hand decisions found in the trace");
    await loadRun(seed, params.get("decision"));
  } catch (error) {
    app.innerHTML = `<div class="loading-card error"><span>LOAD FAILED</span><h1>${esc(error.message)}</h1><p>Start the hand slideshow server and refresh this page.</p></div>`;
  }
}

init();
