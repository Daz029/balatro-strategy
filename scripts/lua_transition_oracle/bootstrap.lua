-- Headless transition harness for Balatro 1.0.1o.
--
-- This file deliberately loads the game's implementation files verbatim.  The
-- definitions below replace only the LÖVE/render/UI/profile persistence shell
-- and CardArea's geometry.  Card, Blind, Back, Event and EventManager are the
-- real classes from the source tree.

local SOURCE = assert(BALATRO_SOURCE, "BALATRO_SOURCE was not supplied")
if oracle_randomseed and oracle_random then
    math.randomseed = oracle_randomseed
    math.random = oracle_random
end

local function source(path)
    local filename = SOURCE .. "/" .. path
    local handle = assert(io.open(filename, "rb"))
    local text = handle:read("*a")
    handle:close()
    local chunk, err = loadstring(text, "@" .. filename)
    assert(chunk, err)
    return chunk()
end

source("engine/object.lua")

-- Minimal drawable objects.  These fields are the ones constructors and
-- gameplay methods inspect; all rendering methods are inert.
Moveable = Object:extend()
local node_id = 0
function Moveable:init(x, y, w, h)
    node_id = node_id + 1
    self.ID = node_id
    self.T = {x = x or 0, y = y or 0, w = w or 1, h = h or 1, r = 0, scale = 1}
    self.VT = {x = self.T.x, y = self.T.y, w = self.T.w, h = self.T.h, r = 0, scale = 1}
    self.states = {
        collide = {can = false}, hover = {can = false}, drag = {can = false, is = false},
        click = {can = false}, visible = true,
    }
    self.children = {}
    self.pinch = {x = false, y = false}
    self.config = self.config or {}
    self.role = {}
end
function Moveable:set_role(config) self.role = config or {} end
function Moveable:hard_set_T() end
function Moveable:hard_set_VT() end
function Moveable:recalculate() end
function Moveable:remove() self.REMOVED = true end
function Moveable:juice_up() end
function Moveable:move() end

Sprite = Moveable:extend()
function Sprite:init(x, y, w, h, atlas, pos)
    Moveable.init(self, x, y, w, h)
    self.atlas, self.sprite_pos = atlas or {}, pos or {x = 0, y = 0}
    self.scale = {x = 1, y = 1}
end
function Sprite:set_sprite_pos(pos) self.sprite_pos = pos end
function Sprite:define_draw_steps() end
function Sprite:rescale() end
function Sprite:reset() end
AnimatedSprite = Sprite
Particles = Sprite

local function null_ui()
    local n = Moveable(0, 0, 1, 1)
    n.alignment = {offset = {x = 0, y = 0}}
    n.config = {object = n}
    n.parent = n
    n.UIRoot = n
    n.children = {n}
    function n:get_UIE_by_ID() return self end
    function n:add_child() end
    function n:pop_in() end
    function n:pop_out() end
    function n:update() end
    function n:remove() self.REMOVED = true end
    return n
end

function UIBox() return null_ui() end
function DynaText() return null_ui() end
function Card_Character() return null_ui() end

love = setmetatable({}, {__index = function()
    return setmetatable({}, {__index = function() return function() return nil end end})
end})

G = {
    TIMERS = {REAL = 0, TOTAL = 0},
    SETTINGS = {paused = false, GAMESPEED = 1, reduced_motion = true, profile = 1,
        tutorial_complete = true},
    ARGS = {spin = {}}, FUNCS = {}, I = {CARD = {}, CARDAREA = {}, SPRITE = {}},
    ROOM = {T = {x = 0, y = 0, w = 20, h = 12}, jiggle = 0},
    C = setmetatable({
        BLACK = {0, 0, 0, 1}, RED = {1, 0, 0, 1}, WHITE = {1, 1, 1, 1},
        BLUE = {0, 0, 1, 1}, GREEN = {0, 1, 0, 1}, MONEY = {1, 1, 0, 1},
        IMPORTANT = {1, 1, 1, 1}, FILTER = {1, 1, 1, 1}, MULT = {1, 0, 0, 1},
        CHIPS = {0, 0, 1, 1}, DARK_EDITION = {1, 1, 1, 1}, EDITION = {1, 1, 1, 1},
        CLEAR = {0, 0, 0, 0}, SUITS = {}, SECONDARY_SET = {}, UI = {}, DYN_UI = {},
    }, {__index = function() return {1, 1, 1, 1} end}),
    ASSET_ATLAS = setmetatable({}, {__index = function() return {} end}),
    ANIMATION_ATLAS = setmetatable({}, {__index = function() return {} end}),
    CARD_W = 1, CARD_H = 1, TILE_W = 20, TILE_H = 12,
    UIT = {ROOT = 1, R = 2, C = 3, O = 4, T = 5},
    STATES = {SELECTING_HAND = 1, HAND_PLAYED = 2, DRAW_TO_HAND = 3,
        ROUND_EVAL = 4, GAME_OVER = 5, NEW_ROUND = 6, SHOP = 7,
        TAROT_PACK = 8, SPECTRAL_PACK = 9},
    STATE = 1, STATE_COMPLETE = true, VIBRATION = 0,
    CONTROLLER = {interrupt = {}, locks = {}, focused = {}},
    PROFILES = {[1] = {career_stats = {c_round_interest_cap_streak = 0},
        high_scores = {current_streak = {amt = 0}}, hand_usage = {},
        challenge_progress = {completed = {}}, progress = {}}},
    FILE_HANDLER = {}, LANGUAGES = {['en-us'] = {font = {}}},
}
G.C.UI.TEXT_LIGHT, G.C.UI.BACKGROUND_INACTIVE = G.C.WHITE, G.C.BLACK
G.C.DYN_UI.MAIN, G.C.DYN_UI.DARK = G.C.BLACK, G.C.BLACK
G.C.DYN_UI.BOSS_MAIN, G.C.DYN_UI.BOSS_DARK = G.C.BLACK, G.C.BLACK
G.C.SECONDARY_SET.Planet = G.C.BLUE

function G.CONTROLLER:save_cardarea_focus() end
function G:save_settings() end
function G:save_progress() end

G.HUD, G.HUD_blind, G.round_eval = null_ui(), null_ui(), null_ui()
G.hand_text_area = {ante = null_ui(), round = null_ui()}

source("engine/event.lua")
source("functions/misc_functions.lua")
source("functions/common_events.lua")
source("functions/state_events.lua")
source("functions/button_callbacks.lua")
source("card.lua")
source("blind.lua")
source("back.lua")
source("game.lua")

-- UI/profile-only replacements.  update_hand_text retains the small state
-- mirror used by evaluate_play itself; the original otherwise only animates it.
function play_sound() end
function attention_text() end
function card_eval_status_text(_, _, _, _, _, extra)
    -- The source function is presentation except for this notification tail.
    if extra and extra.playing_cards_created then
        playing_card_joker_effects(extra.playing_cards_created)
    end
end
function play_area_status_text() end
function juice_card() end
function highlight_card() end
function stop_use() end
function check_for_unlock() end
function check_and_set_high_score() end
function inc_career_stat() end
function inc_steam_stat() end
function discover_card() end
function unlock_card() end
function set_joker_usage() end
function save_run() end
function ease_background_colour_blind() end
function localize(value)
    if value == '$' then return '$' end
    if type(value) == 'string' then return value end
    if type(value) == 'table' and value.type == 'raw_descriptions' then
        return {value.key or 'loc'}
    end
    return (type(value) == 'table' and (value.key or value.type)) or 'loc'
end
function number_format(value) return tostring(value or 0) end
function scale_number(_, scale) return scale or 1 end
function get_compressed() return 'return {}' end
function convert_save_to_meta() end
function STR_UNPACK(text)
    local chunk = assert(loadstring(text))
    return chunk()
end
function darken(colour) return colour end
function lighten(colour) return colour end
function mix_colours(a) return a end
function ease_colour(ref, colour)
    if type(ref) == 'table' and type(colour) == 'table' then
        for k, v in pairs(colour) do ref[k] = v end
    end
end
function update_hand_text(_, values)
    if not (G.GAME and values) then return end
    local hand = G.GAME.current_round.current_hand
    for k, v in pairs(values) do if v ~= nil then hand[k] = v end end
end

-- Build the exact source prototype tables and the source game-state defaults.
Game.init_item_prototypes(G)
for key, value in pairs(G.P_CENTERS) do value.key = key end
for key, value in pairs(G.P_CARDS) do value.key = key end
for key, value in pairs(G.P_BLINDS) do value.key = key end

-- Geometry-free CardArea implementing cardarea.lua's observable ordering.
CardArea = Object:extend()
function CardArea:init(config)
    self.cards, self.highlighted = {}, {}
    self.config = config or {}
    self.config.card_limit = self.config.card_limit or 52
    self.config.highlighted_limit = self.config.highlighted_limit or 5
    self.config.type = self.config.type or 'deck'
    self.config.sort = self.config.sort or 'desc'
    self.T = {x = 0, y = 0, w = 6, h = 1}
end
function CardArea:set_ranks()
    for i, card in ipairs(self.cards) do card.rank = i; card.T.x = i end
end
function CardArea:align_cards()
    if self.config.type == 'deck' or self.config.type == 'discard' then
        for _, card in ipairs(self.cards) do if card.facing == 'front' then card:flip() end end
    end
    self:set_ranks()
end
function CardArea:hard_set_T() end
function CardArea:hard_set_VT() end
function CardArea:change_size(delta) self.config.card_limit = self.config.card_limit + delta end
function CardArea:emplace(card, location, stay_flipped)
    if location == 'front' or self.config.type == 'deck' then table.insert(self.cards, 1, card)
    else self.cards[#self.cards + 1] = card end
    if card.facing == 'back' and self.config.type ~= 'discard' and self.config.type ~= 'deck' and not stay_flipped then
        card:flip()
    end
    if self == G.hand and stay_flipped then card.ability.wheel_flipped = true end
    card:set_card_area(self)
    self:set_ranks()
    self:align_cards()
end
function CardArea:remove_from_highlighted(card)
    for i = #self.highlighted, 1, -1 do
        if self.highlighted[i] == card then table.remove(self.highlighted, i) end
    end
    if card then card.highlighted = false end
end
function CardArea:remove_card(card, discarded_only)
    local candidates = self.cards
    if discarded_only then
        candidates = {}
        for _, candidate in ipairs(self.cards) do
            if candidate.ability.discarded then candidates[#candidates + 1] = candidate end
        end
    end
    if self.config.type == 'discard' or self.config.type == 'deck' then card = card or candidates[#candidates]
    else card = card or candidates[1] end
    for i = #self.cards, 1, -1 do
        if self.cards[i] == card then
            card:remove_from_area()
            table.remove(self.cards, i)
            self:remove_from_highlighted(card)
            self:set_ranks()
            return card
        end
    end
end
function CardArea:draw_card_from(from, stay_flipped, discarded_only)
    local card = from:remove_card(nil, discarded_only)
    if not card then return false end
    stay_flipped = G.GAME and G.GAME.blind and G.GAME.blind:stay_flipped(self, card)
    if self == G.hand and G.GAME.modifiers.flipped_cards then
        if pseudorandom(pseudoseed('flipped_card')) < 1 / G.GAME.modifiers.flipped_cards then
            stay_flipped = true
        end
    end
    self:emplace(card, nil, stay_flipped)
    return true
end
function CardArea:add_to_highlighted(card)
    if #self.highlighted < self.config.highlighted_limit then
        self.highlighted[#self.highlighted + 1] = card
        card.highlighted = true
    end
end
function CardArea:unhighlight_all()
    for _, card in ipairs(self.highlighted) do card.highlighted = false end
    self.highlighted = {}
end
function CardArea:shuffle(seed) pseudoshuffle(self.cards, pseudoseed(seed)); self:set_ranks() end
function CardArea:sort(method)
    self.config.sort = method or self.config.sort
    if self.config.sort == 'desc' then
        table.sort(self.cards, function(a, b) return a:get_nominal() > b:get_nominal() end)
    elseif self.config.sort == 'asc' then
        table.sort(self.cards, function(a, b) return a:get_nominal() < b:get_nominal() end)
    elseif self.config.sort == 'suit desc' then
        table.sort(self.cards, function(a, b) return a:get_nominal('suit') > b:get_nominal('suit') end)
    elseif self.config.sort == 'suit asc' then
        table.sort(self.cards, function(a, b) return a:get_nominal('suit') < b:get_nominal('suit') end)
    elseif self.config.sort == 'order' then
        table.sort(self.cards, function(a, b)
            return (a.config.card.order or a.config.center.order) < (b.config.card.order or b.config.center.order)
        end)
    end
    self:set_ranks()
end

local function overlay(dst, src)
    if type(src) ~= 'table' then return end
    for key, value in pairs(src) do
        if type(value) == 'table' and type(dst[key]) == 'table' then overlay(dst[key], value)
        else dst[key] = value end
    end
end

local observations = nil
local tracing = false
local round_eval_rows = nil

local source_add_round_eval_row = add_round_eval_row
function add_round_eval_row(config)
    config = config or {}
    round_eval_rows[#round_eval_rows + 1] = {
        type = config.name,
        amount = config.dollars or 1,
    }
    return source_add_round_eval_row(config)
end

local function snapshot_card(card)
    local ability = {}
    for key, value in pairs(card.ability or {}) do
        if type(value) ~= 'function' and type(value) ~= 'userdata' then ability[key] = copy_table(value) end
    end
    ability.effect = ability.effect or ''
    return {
        id = card._oracle_id, key = card.config and (card.config.center_key or card.config.card_key),
        base = card.base and card.base.value and {rank = card.base.value, suit = card.base.suit} or nil,
        ability = ability, edition = copy_table(card.edition), seal = card.seal,
        debuff = not not card.debuff, facing = card.facing,
    }
end

local area_names = {'hand', 'play', 'deck', 'discard', 'jokers', 'consumeables'}
local state_names = {
    [G.STATES.SELECTING_HAND] = 'selecting_hand', [G.STATES.ROUND_EVAL] = 'round_eval',
    [G.STATES.GAME_OVER] = 'game_over', [G.STATES.SHOP] = 'shop',
    [G.STATES.DRAW_TO_HAND] = 'draw_to_hand', [G.STATES.NEW_ROUND] = 'new_round',
}
local function snapshot(label, kind)
    local game, round = G.GAME, G.GAME.current_round
    local result = {
        label = label, kind = kind, phase = state_names[G.STATE] or tostring(G.STATE), dollars = game.dollars,
        dollar_buffer = game.dollar_buffer or 0, joker_buffer = game.joker_buffer or 0,
        consumeable_buffer = game.consumeable_buffer or 0, chips = game.chips,
        current_round = {
            hands_left = round.hands_left, discards_left = round.discards_left,
            hands_played = round.hands_played, discards_used = round.discards_used,
            most_played_poker_hand = round.most_played_poker_hand, dollars = round.dollars,
        },
        hands_played = game.hands_played, last_hand_played = game.last_hand_played,
        hands = {}, blind = {
            name = game.blind and game.blind.name, chips = game.blind and game.blind.chips,
            triggered = game.blind and not not game.blind.triggered,
            disabled = game.blind and not not game.blind.disabled,
            prepped = game.blind and not not game.blind.prepped,
        },
        areas = {}, pseudorandom = {}, events_pending = 0,
        round_eval_rows = copy_table(round_eval_rows),
    }
    for key, hand in pairs(game.hands) do
        result.hands[key] = {played = hand.played, played_this_round = hand.played_this_round, level = hand.level}
    end
    for _, name in ipairs(area_names) do
        result.areas[name] = {}
        for i, card in ipairs(G[name].cards) do result.areas[name][i] = snapshot_card(card) end
    end
    for key, value in pairs(game.pseudorandom) do
        if key ~= 'seed' and key ~= 'hashed_seed' then result.pseudorandom[key] = value end
    end
    for _, queue in pairs(G.E_MANAGER.queues) do result.events_pending = result.events_pending + #queue end
    return result
end

local original_event_init = Event.init
function Event:init(config)
    original_event_init(self, config)
    local info = debug.getinfo(config.func or self.func, 'S')
    self._oracle_label = info and (info.short_src .. ':' .. tostring(info.linedefined)) or 'event'
end
local original_event_handle = Event.handle
function Event:handle(results)
    local incomplete = not self.complete
    original_event_handle(self, results)
    if tracing and incomplete and self.complete then
        observations[#observations + 1] = snapshot(self._oracle_label or ('event-' .. #observations), 'event')
    end
end

local function area(type_name, limit)
    return CardArea({type = type_name, card_limit = limit, highlighted_limit = 5})
end

local function make_card(spec, playing_index)
    local front = spec.front and assert(G.P_CARDS[spec.front], 'unknown front ' .. spec.front) or G.P_CARDS.empty
    local center_key = spec.center or spec.key or (spec.front and 'c_base')
    local center = assert(G.P_CENTERS[center_key], 'unknown center ' .. tostring(center_key))
    local card = Card(0, 0, G.CARD_W, G.CARD_H, front, center, {
        playing_card = playing_index, bypass_discovery_center = true,
        bypass_discovery_ui = true,
    })
    card._oracle_id = assert(spec.id, 'every scenario card requires an id')
    if spec.edition then
        local edition = type(spec.edition) == 'string' and {[spec.edition] = true} or spec.edition
        card:set_edition(edition, true, true)
    end
    if spec.seal then card:set_seal(spec.seal, true, true) end
    overlay(card.ability, spec.ability)
    card.debuff = not not spec.debuff
    card.facing = spec.facing or 'front'
    return card
end

local function place_specs(target, specs, playing)
    if not specs then return end
    local first, last, step = 1, #specs, 1
    if target.config.type == 'deck' then first, last, step = #specs, 1, -1 end
    for i = first, last, step do
        local card = make_card(specs[i], playing and (#G.playing_cards + 1) or nil)
        if playing then G.playing_cards[#G.playing_cards + 1] = card end
        target:emplace(card, target.config.type == 'deck' and 'front' or nil, card.facing == 'back')
        if card.ability.set == 'Joker' or card.ability.consumeable then card:add_to_deck() end
    end
end

local function reset(scenario)
    G.TIMERS.REAL, G.TIMERS.TOTAL = 0, 0
    G.STATE, G.STATE_COMPLETE = G.STATES.SELECTING_HAND, true
    G.E_MANAGER = EventManager()
    G.round_eval = null_ui()
    round_eval_rows = {}
    G.GAME = Game.init_game_object(G)
    G.GAME.selected_back = {pos = G.P_CENTERS.b_red.pos}
    G.GAME.viewed_back = G.GAME.selected_back
    G.GAME.seeded = true
    G.GAME.pseudorandom.seed = scenario.seed or 'ORACLE'
    G.GAME.pseudorandom.hashed_seed = pseudohash(G.GAME.pseudorandom.seed)
    G.GAME.stake = scenario.stake or 1
    G.GAME.round_resets.ante = scenario.ante or 1
    G.GAME.round_resets.hands = scenario.hands_per_round or 4
    G.GAME.round_resets.discards = scenario.discards_per_round or 3
    G.GAME.round = scenario.round or 1
    G.GAME.dollars = scenario.dollars or 4
    G.GAME.dollar_buffer = 0
    G.GAME.chips = scenario.chips or 0
    G.GAME.hands_played = scenario.hands_played or 0
    G.GAME.current_round.hands_left = scenario.hands_left or 4
    G.GAME.current_round.discards_left = scenario.discards_left or 3
    G.GAME.current_round.hands_played = scenario.round_hands_played or 0
    G.GAME.current_round.discards_used = scenario.discards_used or 0
    G.GAME.current_round.dollars = scenario.round_dollars or 0
    G.GAME.last_hand_played = scenario.last_hand_played
    overlay(G.GAME.hands, scenario.hand_levels)
    G.GAME.current_round.most_played_poker_hand = scenario.most_played_poker_hand or 'High Card'
    G.GAME.modifiers = scenario.modifiers or {}

    G.hand, G.play = area('hand', scenario.hand_limit or 8), area('play', 5)
    G.deck, G.discard = area('deck', 500), area('discard', 500)
    G.jokers, G.consumeables = area('joker', scenario.joker_limit or 5), area('consumeable', scenario.consumable_limit or 2)
    G.playing_cards = {}

    local areas = scenario.areas or {}
    place_specs(G.hand, areas.hand, true)
    place_specs(G.play, areas.play, true)
    place_specs(G.deck, areas.deck, true)
    place_specs(G.discard, areas.discard, true)
    place_specs(G.jokers, scenario.jokers or areas.jokers, false)
    place_specs(G.consumeables, scenario.consumables or areas.consumeables, false)

    local back_key = scenario.back or 'b_red'
    G.GAME.selected_back = Back(assert(G.P_CENTERS[back_key], 'unknown back ' .. back_key))

    local blind_spec = scenario.blind or {key = 'bl_small'}
    local blind_proto = assert(G.P_BLINDS[blind_spec.key or 'bl_small'], 'unknown blind')
    local blind = Blind(0, 0, 1, 1)
    blind.config.blind, blind.name, blind.dollars = blind_proto, blind_proto.name, blind_proto.dollars
    blind.debuff, blind.pos, blind.mult = copy_table(blind_proto.debuff or {}), blind_proto.pos, blind_proto.mult
    blind.boss, blind.disabled, blind.triggered = not not blind_proto.boss, false, false
    blind.prepped, blind.hands, blind.only_hand = false, {}, false
    blind.chips = get_blind_amount(G.GAME.round_resets.ante) * blind.mult * (G.GAME.starting_params.ante_scaling or 1)
    blind.chip_text = tostring(blind.chips)
    overlay(blind, blind_spec.overrides)
    G.GAME.blind, G.GAME.round_resets.blind = blind, blind_proto
    G.GAME.round_resets.blind_choices.Boss = blind_spec.key
    G.GAME.blind_on_deck = blind.name == 'Small Blind' and 'Small' or (blind.name == 'Big Blind' and 'Big' or 'Boss')
    return scenario
end

local function action_name(action) return action.type or action.action end
local function run_action(scenario)
    local action = assert(scenario.action, 'scenario.action is required')
    local name = assert(action_name(action), 'scenario.action.type is required')
    if name == 'play' then
        G.STATE = G.STATES.HAND_PLAYED
        local picked = {}
        for _, index in ipairs(action.indices or {}) do picked[#picked + 1] = G.hand.cards[index + 1] end
        table.sort(picked, function(a, b) return a.T.x < b.T.x end)
        ease_hands_played(-1)
        for _, card in ipairs(picked) do
            card.base.times_played = card.base.times_played + 1
            card.ability.played_this_ante = true
            G.hand:remove_card(card)
            G.play:emplace(card)
        end
        G.GAME.blind.triggered = false
        G.GAME.blind:press_play()
        G.FUNCS.evaluate_play()
        G.E_MANAGER:add_event(Event({trigger = 'after', delay = 0.1, func = function()
            G.FUNCS.draw_from_play_to_discard()
            G.GAME.hands_played = G.GAME.hands_played + 1
            G.GAME.current_round.hands_played = G.GAME.current_round.hands_played + 1
            return true
        end}))
    elseif name == 'discard' then
        G.hand:unhighlight_all()
        for _, index in ipairs(action.indices or {}) do G.hand:add_to_highlighted(G.hand.cards[index + 1], true) end
        G.FUNCS.discard_cards_from_highlighted(nil, false)
        G.E_MANAGER:add_event(Event({trigger = 'after', delay = 0.5, func = function()
            G.FUNCS.draw_from_deck_to_hand()
            G.E_MANAGER:add_event(Event({trigger = 'after', delay = 0.5, func = function()
                G.GAME.blind:drawn_to_hand()
                return true
            end}))
            return true
        end}))
    elseif name == 'end_round' then
        end_round()
    elseif name == 'cash_out' then
        G.round_eval = null_ui()
        G.FUNCS.cash_out({config = {button = 'cash_out'}})
    elseif name == 'set_blind' then
        local key = action.key or (scenario.blind and scenario.blind.key)
        G.GAME.round_resets.blind = assert(G.P_BLINDS[key], 'unknown blind ' .. tostring(key))
        G.GAME.round_resets.blind_choices.Boss = key
        new_round()
        G.E_MANAGER:add_event(Event({trigger = 'after', delay = 1.0, func = function()
            G.FUNCS.draw_from_deck_to_hand()
            G.E_MANAGER:add_event(Event({trigger = 'after', delay = 0.5, func = function()
                G.GAME.blind:drawn_to_hand()
                return true
            end}))
            return true
        end}))
    else
        error('unknown oracle action ' .. tostring(name))
    end
end

local function pending_events()
    local count = 0
    for _, queue in pairs(G.E_MANAGER.queues) do count = count + #queue end
    return count
end

local function drain()
    local cap, dt = 20000, 1 / 60
    for iteration = 1, cap do
        if pending_events() == 0 then return iteration - 1 end
        G.TIMERS.REAL = G.TIMERS.REAL + dt
        G.TIMERS.TOTAL = G.TIMERS.TOTAL + dt
        G.E_MANAGER:update(dt, true)
    end
    error('EventManager drain exceeded ' .. cap .. ' updates with ' .. pending_events() .. ' events pending')
end

local function escape(value)
    return value:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
end
local function is_array(value)
    local maximum, count = 0, 0
    for key in pairs(value) do
        if type(key) ~= 'number' or key < 1 or key ~= math.floor(key) then return false end
        maximum, count = math.max(maximum, key), count + 1
    end
    return maximum == count
end
local function json(value, seen)
    local kind = type(value)
    if kind == 'nil' then return 'null' end
    if kind == 'boolean' then return value and 'true' or 'false' end
    if kind == 'number' then return tostring(value) end
    if kind == 'string' then return '"' .. escape(value) .. '"' end
    if kind ~= 'table' then return 'null' end
    seen = seen or {}
    if seen[value] then return 'null' end
    seen[value] = true
    local parts = {}
    if is_array(value) then
        for i = 1, #value do parts[#parts + 1] = json(value[i], seen) end
        seen[value] = nil
        return '[' .. table.concat(parts, ',') .. ']'
    end
    for key, item in pairs(value) do
        if type(key) == 'string' then parts[#parts + 1] = json(key) .. ':' .. json(item, seen) end
    end
    table.sort(parts)
    seen[value] = nil
    return '{' .. table.concat(parts, ',') .. '}'
end

function run_transition_oracle(scenario)
    tracing = false
    reset(scenario)
    observations = {}
    tracing = true
    run_action(scenario)
    table.insert(observations, 1, snapshot('synchronous', 'synchronous'))
    local updates = drain()
    if action_name(scenario.action) == 'play' then
        if G.GAME.chips >= G.GAME.blind.chips or G.GAME.current_round.hands_left <= 0 then
            end_round()
            updates = updates + drain()
            if G.STATE == G.STATES.ROUND_EVAL then
                G.FUNCS.evaluate_round()
                updates = updates + drain()
            end
        else
            G.STATE = G.STATES.DRAW_TO_HAND
            G.FUNCS.draw_from_deck_to_hand()
            updates = updates + drain()
            G.GAME.blind:drawn_to_hand()
            G.STATE = G.STATES.SELECTING_HAND
            updates = updates + drain()
        end
    end
    if action_name(scenario.action) == 'end_round' and G.STATE == G.STATES.ROUND_EVAL then
        G.FUNCS.evaluate_round()
        updates = updates + drain()
    end
    observations[#observations + 1] = snapshot('drained', 'drained')
    tracing = false
    return json({observations = observations, drain_updates = updates, source = SOURCE})
end
