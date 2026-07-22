# Hand decision slideshow

This viewer presents the hand decisions in the h2 PPO evaluation trace using
the same Balatro-inspired visual language as the shop slideshow.

From the repository root:

```powershell
.\.venv\Scripts\python.exe prototypes/hand-slideshow/server.py
```

Then open <http://127.0.0.1:8766/>.

The default trace is
`runs/hand_ppo_b/h2/dumped_hand_eval.jsonl`. Use another compatible dump with:

```powershell
.\.venv\Scripts\python.exe prototypes/hand-slideshow/server.py --trace path/to/trace.jsonl
```

Use the buttons, left/right arrows, or Space/Shift+Space to step through a
run. The bottom-right button (or `O`) reveals the recorded order in which the
model selected its cards. Hands are displayed rank-first from Ace down, with
Spades, Hearts, Clubs, then Diamonds as the suit tie-breaker.
