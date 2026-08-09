# Shop slideshow prototype

Question: what should a frame-by-frame, Balatro-like shop policy viewer look
and feel like?

The prototype started with three variants. The selected direction is Decision
Desk (formerly variant B): a compact board that keeps the full shop state
visible at once.

The final adjustments are:

- Reroll and Next Blind live in a dedicated action rail on the left.
- Jokers, consumables, shop cards, boosters, voucher, and HUD occupy the larger
  board to the right.
- The Decision Forensics panel and prototype variant switcher are removed.
- The model's selected card or action has a solid yellow outline.
- Blind-selection frames are included; the left rail shows the engine-derived
  score requirement and a Skip Blind button so skip decisions are visible.

Run with one command from the repository root:

```powershell
python prototypes/shop-slideshow-prototype/server.py
```

Then open <http://127.0.0.1:8765/>.

Prototype verdict: **Decision Desk selected.** These layout decisions are ready
to carry into a maintained viewer when the prototype is promoted.
