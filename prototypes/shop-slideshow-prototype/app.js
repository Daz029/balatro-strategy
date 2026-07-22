/* PROTOTYPE: the selected Decision Desk shop-slideshow layout. */

const app = document.querySelector("#app");

const params = new URLSearchParams(window.location.search);
const state = {
  runs: [],
  seed: params.get("seed"),
  frames: [],
  frameIndex: 0,
};

const esc = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const setClass = (value) => String(value || "card").toLowerCase().replaceAll(" ", "-");

function targetMatches(frame, kind, slot) {
  const target = frame.target;
  if (!target) return false;
  if (target.kind === "target_cards" && kind === "target_card") {
    return (target.slots || []).map(Number).includes(slot);
  }
  return target.kind === kind && Number(target.slot) === slot;
}

function cardGlyph(card) {
  return {
    Joker: "★",
    Tarot: "✦",
    Planet: "●",
    Spectral: "◆",
    Voucher: "V",
    Booster: "PACK",
    Default: "♠",
  }[card?.set] || "◆";
}

function cardHTML(card, slot, kind, options = {}) {
  if (!card) {
    return `<div class="game-card empty-card ${options.mini ? "mini" : ""}">
      <span>EMPTY</span><small>SLOT ${slot + 1}</small>
    </div>`;
  }
  const picked = targetMatches(options.frame, kind, slot);
  const flags = [card.edition, card.eternal && "eternal", card.rental && "rental"]
    .filter(Boolean)
    .join(" · ");
  const cost = card.cost === null || card.cost === undefined ? "" : `<b>$${card.cost}</b>`;
  return `<article class="game-card set-${setClass(card.set)} ${picked ? "picked" : ""} ${
    options.mini ? "mini" : ""
  } ${options.hero ? "hero-card" : ""}" title="${esc(card.key || card.name)}">
    ${picked ? '<span class="pick-ribbon">MODEL PICK</span>' : ""}
    <span class="card-glyph">${esc(cardGlyph(card))}</span>
    <div class="card-copy">
      <strong>${esc(card.name)}</strong>
      <small>${esc(card.detail || card.set)}</small>
    </div>
    <div class="card-foot"><span>${esc(flags)}</span>${cost}</div>
  </article>`;
}

function slotsHTML(cards, max, kind, frame, options = {}) {
  return Array.from({ length: max }, (_, slot) =>
    cardHTML(cards[slot], slot, kind, { ...options, frame }),
  ).join("");
}

function actionCopy(frame) {
  const picked = frame.target?.card?.name;
  const selectedCount = frame.target?.cards?.length || 0;
  const labels = {
    BuyCard: picked ? `Buy ${picked}` : "Buy card",
    OpenBooster: picked ? `Open ${picked}` : "Open booster",
    PickPackCard: picked ? `Take ${picked}` : "Take pack card",
    SkipPack: "Skip this pack",
    Reroll: `Reroll shop for $${frame.reroll_cost}`,
    NextRound:
      frame.phase === "blind_select"
        ? `Play ${frame.upcoming_blind.name}`
        : `Leave shop · ${frame.upcoming_blind.name}`,
    SkipBlind: `Skip ${frame.upcoming_blind.name}`,
    SellJoker: picked ? `Sell ${picked}` : "Sell joker",
    SellJokerExt: picked ? `Sell ${picked}` : "Sell joker",
    SellConsumable: picked ? `Sell ${picked}` : "Sell consumable",
    UseConsumable: picked ? `Use ${picked}` : "Use consumable",
    RedeemVoucher: picked ? `Redeem ${picked}` : "Redeem voucher",
    SelectTarget: `Select ${selectedCount} target card${selectedCount === 1 ? "" : "s"}`,
  };
  return labels[frame.action] || frame.action_label || frame.action;
}

function tagName(key) {
  if (!key) return "NO SKIP AVAILABLE";
  return key.replace(/^tag_/, "").replaceAll("_", " ").toUpperCase();
}

function updateUrl() {
  const frame = state.frames[state.frameIndex];
  const next = new URLSearchParams(window.location.search);
  next.delete("variant");
  if (state.seed) next.set("seed", state.seed);
  if (frame) next.set("step", frame.step);
  window.history.replaceState({}, "", `${window.location.pathname}?${next}`);
}

function controlsHTML(frame) {
  const options = state.runs
    .map(
      (run) =>
        `<option value="${esc(run.seed)}" ${run.seed === state.seed ? "selected" : ""}>${esc(
          run.seed,
        )} · ${run.frames} frames</option>`,
    )
    .join("");
  return `<header class="transport">
    <div class="transport-brand"><span>POLICY</span><strong>SHOP REPLAY</strong></div>
    <label class="seed-control">GAME<select data-control="seed">${options}</select></label>
    <div class="step-controls">
      <button data-action="previous" ${state.frameIndex === 0 ? "disabled" : ""}>◀</button>
      <button class="step-button" data-action="next" ${
        state.frameIndex === state.frames.length - 1 ? "disabled" : ""
      }>STEP ▶</button>
      <label>TIME STEP <input id="step-input" type="number" min="1" value="${esc(
        frame.step,
      )}" /></label>
      <button data-action="jump">GO</button>
    </div>
    <div class="frame-count"><b>${state.frameIndex + 1}</b> / ${state.frames.length}<small>SPACE TO STEP</small></div>
  </header>`;
}

function inventoryHTML(frame, compact = false) {
  return `<section class="inventory-top ${compact ? "compact" : ""}">
    <div class="rack-label"><b>JOKERS</b><span>${frame.jokers.length} / ${frame.joker_slots}</span></div>
    <div class="joker-rack">${slotsHTML(frame.jokers, 8, "joker", frame, { mini: compact })}</div>
    <div class="consumable-zone">
      <div class="rack-label"><b>CONSUMABLES</b><span>${frame.consumables.length} / ${
        frame.consumable_slots
      }</span></div>
      <div class="consumable-rack">${slotsHTML(
        frame.consumables,
        Math.max(2, frame.consumable_slots),
        "consumable",
        frame,
        { mini: true },
      )}</div>
    </div>
  </section>`;
}

function actionBanner(frame, label = "MODEL DECISION") {
  const delta = frame.money_delta
    ? `<span class="delta ${frame.money_delta > 0 ? "positive" : "negative"}">${
        frame.money_delta > 0 ? "+" : ""
      }$${frame.money_delta}</span>`
    : "";
  return `<div class="action-banner">
    <small>${label} · STEP ${frame.step}</small>
    <strong>${esc(actionCopy(frame))}</strong>${delta}
  </div>`;
}

function packStageHTML(frame, focus = false) {
  const showTargets = frame.action === "SelectTarget" && frame.pack_hand.length;
  const cards = showTargets ? frame.pack_hand : frame.pack_cards;
  const kind = showTargets ? "target_card" : "pack_card";
  const count = Math.max(cards.length, 1);
  return `<section class="pack-stage ${focus ? "focus" : ""}">
    <div class="pack-heading">
      <span class="eyebrow">${esc(frame.pack_type || "BOOSTER OPENED")}</span>
      <h2>${showTargets ? "Choose the target" : "Choose a card"}</h2>
      <p>${
        frame.pack_choices_remaining ?? "—"
      } choice remaining · the glowing card is the model's action</p>
    </div>
    <div class="pack-card-row" style="--pack-count:${count}">${cards
      .map((card, slot) => cardHTML(card, slot, kind, { frame, hero: focus }))
      .join("")}</div>
    <button class="skip-pack ${frame.action === "SkipPack" ? "picked-button" : ""}">SKIP PACK</button>
    ${actionBanner(frame, "PACK DECISION")}
  </section>`;
}

function miniZone(title, cards, max, kind, frame) {
  return `<section class="desk-zone">
    <h3>${title}<span>${cards.length}</span></h3>
    <div class="desk-card-row">${slotsHTML(cards, max, kind, frame, { mini: true })}</div>
  </section>`;
}

function actionRailHTML(frame) {
  const blind = frame.upcoming_blind;
  const inShop = frame.phase === "shop";
  return `<aside class="desk-action-rail">
    <div class="rail-label">BLIND & SHOP ACTIONS</div>
    <button class="desk-action-button reroll ${
      frame.action === "Reroll" ? "model-pick" : ""
    } ${inShop ? "" : "unavailable"}" ${inShop ? "" : "disabled"}>
      <small>REFRESH SHOP</small><strong>REROLL</strong><span>$${frame.reroll_cost}</span>
    </button>
    <div class="score-requirement">
      <small>SCORE REQUIREMENT</small><strong>${Number(blind.chips).toLocaleString()}</strong>
      <span>CHIPS</span>
    </div>
    <button class="desk-action-button next-blind ${
      frame.action === "NextRound" ? "model-pick" : ""
    }">
      <small>PROCEED TO</small><strong>NEXT BLIND</strong><span>${esc(blind.name)}</span>
    </button>
    <button class="desk-action-button skip-blind ${
      frame.action === "SkipBlind" ? "model-pick" : ""
    } ${blind.can_skip ? "" : "unavailable"}" ${blind.can_skip ? "" : "disabled"}>
      <small>${blind.can_skip ? "TAKE TAG" : "BOSS BLIND"}</small><strong>SKIP BLIND</strong>
      <span>${esc(tagName(blind.skip_tag))}</span>
    </button>
    <div class="rail-decision">
      <small>MODEL DECISION · STEP ${frame.step}</small>
      <strong>${esc(actionCopy(frame))}</strong>
    </div>
  </aside>`;
}

function DecisionDesk(frame) {
  const pack = frame.phase === "pack_opening";
  const mode = pack ? "pack-mode" : frame.phase === "blind_select" ? "blind-select-mode" : "shop-mode";
  return `<div class="prototype-shell decision-desk ${mode}">
    ${controlsHTML(frame)}
    <div class="desk-layout">
      ${pack ? "" : actionRailHTML(frame)}
      <div class="desk-board">
        ${inventoryHTML(frame, true)}
        ${
          pack
            ? packStageHTML(frame)
            : `<div class="desk-shop-grid">
                ${miniZone("SHOP", frame.shop_cards, 4, "shop_card", frame)}
                ${miniZone("BOOSTERS", frame.boosters, 2, "booster", frame)}
                ${miniZone("VOUCHER", frame.vouchers, 1, "voucher", frame)}
              </div>
              <div class="desk-hud">$${frame.dollars}<span>ANTE ${frame.ante}</span><span>ROUND ${
                frame.round
              }</span><span>${esc(frame.blind_on_deck || "")}</span></div>`
        }
      </div>
    </div>
  </div>`;
}

function render() {
  const frame = state.frames[state.frameIndex];
  if (!frame) return;
  app.innerHTML = DecisionDesk(frame);
  updateUrl();
}

function moveFrame(delta) {
  state.frameIndex = Math.max(0, Math.min(state.frames.length - 1, state.frameIndex + delta));
  render();
}

function jumpToStep(rawStep) {
  const step = Number(rawStep);
  if (!Number.isFinite(step)) return;
  let best = 0;
  for (let index = 1; index < state.frames.length; index += 1) {
    if (Math.abs(state.frames[index].step - step) < Math.abs(state.frames[best].step - step)) {
      best = index;
    }
  }
  state.frameIndex = best;
  render();
}

async function loadRun(seed, requestedStep = null) {
  app.classList.add("is-loading");
  const response = await fetch(`/api/frames?seed=${encodeURIComponent(seed)}`);
  const payload = await response.json();
  state.seed = seed;
  state.frames = payload.frames || [];
  state.frameIndex = 0;
  if (requestedStep !== null) jumpToStep(requestedStep);
  else render();
  app.classList.remove("is-loading");
}

app.addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const action = button.dataset.action;
  if (action === "next") moveFrame(1);
  if (action === "previous") moveFrame(-1);
  if (action === "jump") jumpToStep(document.querySelector("#step-input")?.value);
});

app.addEventListener("change", (event) => {
  if (event.target.matches('[data-control="seed"]')) loadRun(event.target.value);
});

app.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.matches("#step-input")) {
    jumpToStep(event.target.value);
  }
});

window.addEventListener("keydown", (event) => {
  const editing = event.target.matches("input, select, textarea, [contenteditable]");
  if (editing) return;
  if (event.key === "ArrowRight") moveFrame(1);
  if (event.key === "ArrowLeft") moveFrame(-1);
  if (event.code === "Space") {
    event.preventDefault();
    moveFrame(event.shiftKey ? -1 : 1);
  }
});

async function init() {
  try {
    const response = await fetch("/api/runs");
    state.runs = await response.json();
    const requestedSeed = state.runs.some((run) => run.seed === state.seed)
      ? state.seed
      : state.runs[0]?.seed;
    if (!requestedSeed) throw new Error("No shop frames found in the trace");
    await loadRun(requestedSeed, params.get("step"));
  } catch (error) {
    app.innerHTML = `<div class="loading-card error"><span>LOAD FAILED</span><h1>${esc(
      error.message,
    )}</h1><p>Start the prototype with its Python server and refresh.</p></div>`;
  }
}

init();
