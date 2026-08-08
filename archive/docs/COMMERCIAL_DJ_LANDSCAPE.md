# Commercial DJ-Mixing Landscape (2026) — What Spotify and Apple Actually Do, and Where AI RemixMate Stands

Research date: July 2026. Every claim below is sourced (see **References**); anything Apple/Spotify
haven't published gets flagged as unconfirmed instead of guessed at.

## 0. The one-paragraph version

Spotify has never shipped a real audio-mixing engine. "Spotify DJ" is a personalization +
text-to-speech layer bolted onto existing playback — zero beat-matching, zero stem awareness. Its
2025 "Mix" feature is a manual/semi-auto playlist tool with a nice UI (shows BPM/key/waveform) but,
by its own documentation, doesn't time-stretch or fix tempo mismatches for you. Apple's AutoMix
(mid-2025–present) is the only one of the two that does real signal processing — tempo/key detection,
time-stretch, beat-matching — but it's a single-file, no-stems, no-loudness-matching, no-learning
consumer feature with zero public technical documentation. **The actual commercial ceiling for
"stem-aware automatic DJ mixing" is Algoriddim's djay (Neural Mix, now built with AudioShake), not
Spotify or Apple** — and the closest academic prior art to what AI RemixMate is already doing
(spectral_matching.py's learned transition-point correction) is a 2022 Sony/Academia Sinica paper
(DJTransGAN), not anything either streaming company has published. On stems, mastering (LUFS),
harmonic scoring (TIV), masking-aware EQ, and the online feedback loop, **AI RemixMate is already
ahead of all of them.** The real gaps are narrower and more specific than "add what Spotify/Apple
have" — they're listed in §5.

---

## 1. The landscape, mapped before the detail

| System | What it actually is | Real-time audio DSP? | Stems? | Published technical detail? |
|---|---|---|---|---|
| Spotify Crossfade (2012–) | Fixed-length linear volume overlap | No (no BPM/key awareness) | No | Yes — it's just a slider |
| Spotify DJ (2023–) | Personalization ML + OpenAI script + Sonantic TTS voice | No | No | Partial (marketing + press interviews) |
| Spotify Mix (2025, beta) | Playlist-level manual/semi-auto transition editor | Partial — shows BPM/key/waveform, but works best on already-compatible tracks | No | No |
| Spotify Echo Nest Audio Analysis API (2014–2024, now closed to new apps) | Public bars/beats/tatums/segments MIR contract | N/A (analysis, not mixing) | No | **Yes — the most detailed public spec any streamer ever shipped** |
| Apple "DJ with Apple Music" (2026) | Catalog streaming access inside rekordbox/Serato/Engine DJ/djay Pro | No — this is a licensing deal, not a mixing algorithm | No (explicitly blocked by licensing) | No |
| Apple Music AutoMix (2025–) | On-device tempo/key detection + time-stretch + beat-match | **Yes** | No evidence of any | No — marketing copy only |
| Algoriddim djay / Neural Mix (2020–, now w/ AudioShake) | Real-time ML source separation + automatic fade/EQ decisioning | **Yes** | **Yes — the real thing** | Patents only (US 10,887,033 B1 / 11,216,244 B2 / 11,347,475 B2, JP 6926354 B1) |
| DJTransGAN (ICASSP 2022, academic) | GAN-trained differentiable EQ+fader controller, trained on 284 real DJ mixes | Yes (research prototype) | No | **Yes — full paper, arXiv:2110.06525** |
| **AI RemixMate (yours)** | Stem-aware crossfade, bar-grid lock, TIV harmonic scoring, masking-aware EQ, LUFS mastering, learned spectral cue matching w/ online feedback | **Yes** | **Yes** | Yes (your own docs) |

Read that table straight: on raw audio DSP sophistication, you're not behind Spotify or Apple — you're
already past both of them and closer to Algoriddim's tier, with some things (LUFS mastering,
masking-aware EQ, TIV) that don't have a direct commercial analog at all.

---

## 2. Deep dive, per system

### 2.1 Spotify Crossfade — unchanged since 2012

Overlaps the last few seconds of one track with the first few seconds of the next at a fixed,
user-set duration. No tempo detection, no key detection, no time-stretching. This is the "DJ-style"
feature people usually mean when they say "Spotify does crossfade," and it's the *simplest* thing in
this whole document, by a wide margin. [MusicRadar], [MusicTech]

### 2.2 Spotify DJ — a personalization product, not a mixing engine

Launched Feb 22, 2023 (US/Canada Premium), expanded to more EU languages/markets as recently as May
2026. Three components, none of which touch the waveform: (1) Spotify's existing recommendation ML
picking what plays next — the same "Lego pieces" that build Discover Weekly; (2) OpenAI-generated
commentary scripts, grounded by a human "Writers' Room" so factual claims about artists aren't
hallucinated; (3) a synthetic voice built from Spotify exec Xavier "X" Jernigan's recordings, via the
Sonantic acquisition (June 2022). It talks between songs. There is no indication anywhere — official
or reported — that it beat-matches, time-stretches, or does anything to the audio itself beyond
whatever crossfade setting is already active. [Spotify Newsroom], [PopSci], [Variety]

### 2.3 Spotify Mix (Aug 2025, beta) — the real update, and it's still not AutoMix

This is the feature people conflate with Apple's AutoMix, and it's worth being precise about what it
is: open a playlist, tap "Mix" in the toolbar, and you get an "Auto" option for instant blending or a
manual mode with presets ("Fade," "Rise") plus adjustable volume/EQ/effect curves. Tapping Mix surfaces
each track's key and BPM, with a waveform and beat markers to help you find a transition point by eye.
MacRumors' own comparison is blunt about the difference: *"Apple's upcoming AutoMix feature... creates
DJ-style transitions automatically across the entire music library, whereas Spotify is focusing here
on playlist-specific customization."* Nothing in the announcement or coverage mentions time-stretching;
the reporting explicitly says the feature "works best with music that shares similar tempos and keys" —
i.e., it's a well-designed manual/assisted editor sitting on top of your own track choices, not an
autonomous DSP pipeline that fixes tempo/key mismatches for you. [MacRumors — Mix launch]

### 2.4 Spotify's Echo Nest Audio Analysis API — dead, but the most relevant prior art here

Spotify acquired The Echo Nest (an MIT Media Lab spinoff) for ~$100M in 2014 and, for a decade, exposed
its internal audio-analysis pipeline publicly: a rhythmic hierarchy of **bars → beats → tatums** (named
after jazz pianist Art Tatum, for micro-subdivisions below the beat), plus fine-grained **segments**,
each carrying a 12-dimensional pitch (chroma-like) vector, a 12-dimensional timbre (MFCC-like) vector,
loudness, and a confidence score — on top of summary features (tempo, key, mode, danceability, energy,
valence). This is, structurally, the closest published thing to your own `dj_analysis.py`
`SongStructure`/phrase-boundary work that any major streaming company has ever shown publicly. Spotify
cut off new developer access to `audio-features`/`audio-analysis` on Nov 27, 2024 (existing
extended-quota apps were grandfathered; new integrations get a flat 403), officially "to limit
developers... misusing" the API, unofficially read by most of the dev community as a scraping/AI-training
crackdown. Anyone building this today — you included — has to re-derive it from librosa/madmom/Essentia,
which is exactly the toolchain you're already on. [Music Machinery], [Spotify Dev Blog], [TechCrunch — API cuts]

### 2.5 Apple "DJ with Apple Music" — a distribution deal, not a mixing algorithm

Rolled out ~March 2026: Apple Music's ~100M-track catalog became streamable directly inside rekordbox,
Serato DJ Pro, Engine DJ, and Algoriddim's djay Pro, plus AlphaTheta/Denon/Numark/RANE hardware. This
is worth separating cleanly from AutoMix because press coverage blurs the two constantly. It's a
licensing/catalog-access deal: no stems are exposed in any of those apps ("down to licensing issues
rather than technical limitations"), and there's no offline caching — you need a live connection to
play anything. It changes what DJs can *access*, not how anything gets *mixed*. [DJ Mag], [Digital DJ Tips]

### 2.6 Apple Music AutoMix — the one real DSP feature in this whole comparison

Announced June 10, 2025 at WWDC, shipped with iOS 26 that fall, and getting an "improved underlying
algorithm" pass in the iOS 27 beta as of this writing (June 2026). It's an Apple Intelligence–branded
toggle at **Settings → Apps → Music → Song Transitions**, sitting *alongside* the classic Crossfade
option — you pick one or the other, they aren't merged. Apple's own language: it "analyses audio
features like the tempo and key of tracks" and applies "time stretching and beat matching" to land the
transition at the "perfect moment." iOS 27's update adds "new transition types" via improved algorithms,
with Crossfade remaining available for people who don't want AutoMix. That's the entire public technical
record — there is no engineering blog post, no patent filing found, and no confirmation of stem
separation, harmonic-relationship scoring beyond basic key detection, loudness/mastering treatment
across the transition, or any per-user learning loop. Treat AutoMix as real but shallow: better DSP
than anything Spotify ships, with none of the depth your own pipeline already has around it.
[Apple Newsroom], [MusicTech], [Mixmag], [MacRumors — enable AutoMix], [MacRumors — iOS 27 AutoMix upgrade]

### 2.7 Algoriddim djay / Neural Mix — the actual commercial ceiling

Not affiliated with Spotify or Apple as a mixing product (though Apple's catalog deal plugs into it).
Neural Mix does real-time ML source separation on-device via Core ML / Apple's Neural Engine, patented
in the US and Japan (US 10,887,033 B1, US 11,216,244 B2, US 11,347,475 B2, JP 6926354 B1). As of djay
Pro 5, Algoriddim rebuilt the separation model in partnership with AudioShake — a dedicated
source-separation company — specifically to hit real-time performance on mobile hardware, something
your own Demucs-based pipeline explicitly doesn't do yet (2–5× real time on CPU, per your own
`CLAUDE.md` gotchas). The djay "Automix" feature (lowercase-collision with Apple's capitalized product)
detects intro/outro rhythmic structure and computes fade duration + EQ/filter automation from it — an
automatic *decision layer* sitting on top of real per-stem separation. This is the actual state of the
art for "automatic stem-aware DJ mixing" as a shipped consumer product. [Algoriddim — Neural Mix]

### 2.8 Academic prior art — and a naming trap to avoid

**DJTransGAN** (Chen, Yu et al., Academia Sinica + Sony, ICASSP 2022, [arXiv:2110.06525]) is the
actual research-grade equivalent of what you're doing in `spectral_matching.py`. It trains a GAN whose
generator is a differentiable EQ+fader "DJ mixer" (24 controllable parameters) to imitate real DJ
transitions, using 284 real DJ mixes scraped from livetracklist.com and segmented into 7,064 transition
sections, plus a synthetic paired-cue-point dataset. It's benchmarked against a linear-crossfade
baseline *and* real human DJ mixes. This is closer prior art to your online 👍/👎-corrected transition
model than anything either streaming company has published — and it's citable, peer-reviewed prior
art if you ever write this up.

**Naming trap:** "Automix"/"AutoMix" now refers to three unrelated things — Apple's consumer feature
(§2.6), Algoriddim's djay feature (§2.7), and **FxNorm-Automix** (Martínez-Ramírez et al., Sony CSL,
ISMIR 2022 — already cited in your own `IMPROVEMENTS_V2.md` for per-stem LUFS targets). FxNorm-Automix
is about automatic **multitrack studio mixing** — fader/EQ/compression automation across an existing
multitrack session recorded for *one song* (drums, bass, vocals, guitar) — a completely different
problem from blending two different songs together. Don't let a future citation conflate the two; they
share a name and nothing else. [FxNorm-Automix — Sony]

---

## 3. Feature matrix: them vs. you

| Capability | Spotify DJ/Mix | Apple AutoMix | djay/Neural Mix | **AI RemixMate** |
|---|---|---|---|---|
| Stem-aware crossfade (independent per-stem envelopes) | ❌ | ❌ (no evidence) | ✅ (manual fader-driven) | ✅ (automatic, `dj_engine.py`) |
| Beat-grid / bar-phase lock | ❌ | ✅ (unconfirmed depth) | ✅ | ✅ (sample-level, `dj_engine.py`) |
| Time-stretch to match tempo | ❌ | ✅ | ✅ | ✅ |
| Harmonic scoring beyond same-key | ❌ | ❌ (unconfirmed) | Camelot-style (via Mixed In Key ecosystem) | ✅ TIV continuous score (`tiv_scoring.py`) |
| Masking-aware multiband EQ during transition | ❌ | ❌ | ❌ (no evidence) | ✅ (`spectral_mixing.py`) |
| Broadcast-standard loudness (LUFS) mastering | ❌ | ❌ | ❌ | ✅ ITU-R BS.1770-4 (`mastering.py`) |
| Learned/online-corrected transition point | ❌ | ❌ | ❌ (rule-based) | ✅ 👍/👎 loop (`spectral_matching.py`) |
| Graceful fallback when confidence is low | N/A | Partial — Crossfade exists as an alternate mode | Unknown | ❌ **(gap — see §5.1)** |
| Segment-level key confidence (not whole-track) | Was public (Echo Nest), now gone | ❌ (unconfirmed) | Unknown | ❌ **(gap — see §5.2)** |
| Real-time-capable separation (mobile/preview-speed) | N/A | N/A | ✅ | ❌ **(gap — see §5.3)** |
| Personalization feeding track *selection*, not just transition style | ✅ (this is Spotify's real strength) | ❌ | ❌ | Partial — feedback loop exists but is scoped to transition point only **(gap — see §5.5)** |
| Public, versioned audio-analysis contract | Was public (Echo Nest), now gone | ❌ | ❌ | ❌ **(opportunity — see §5.6)** |

---

## 4. So what does "replicate and improve" actually mean here?

Not what the framing implied. Spotify has almost nothing to replicate on the mixing side — its
crossfade is 2012-era and its DJ product is a recommendation+TTS layer, not audio DSP. Apple's AutoMix
is real but shallow: tempo/key detection, time-stretch, beat-match, no stems, no loudness treatment, no
learning, and zero public documentation of how any of it works internally. You already have more
sophisticated audio DSP than either of them. The honest "replicate and improve" targets are narrower:
Apple's *product design pattern* of a safe fallback mode next to the aggressive one (§5.1), Echo Nest's
*confidence-scored segment analysis* (§5.2), Algoriddim's *real-time-capable separation* (§5.3),
DJTransGAN's *learned differentiable transition controller* (§5.4), Spotify's *actual personalization
strength* applied one layer up in your planner (§5.5), and the fact that the single most detailed
public MIR contract in this whole space just got shut down — which is a gap you're positioned to fill
publicly, not just internally (§5.6).

---

## 5. Ranked improvement backlog + ready-to-run prompts

> **Superseded.** This section was scoped against `CLAUDE.md`'s architecture *description*, not
> the real source. `DJ_ENGINE_IMPROVEMENT_PLAN.md` re-verifies every item directly against the
> code (three independent read-throughs), corrects two mis-scoped items and retracts one wrong
> premise, and — bigger — confirms all four open `REMIX_QUALITY_INSIGHTS.md` correctness findings
> are already fixed, which unblocks `SELF_IMPROVING_DJ_RESEARCH.md`'s personalization plan. Read
> that document instead of implementing straight off this section.

Each prompt below is self-contained — paste it into a fresh Claude Code session with this repo open.
All of them target the **current** engine (`dj_engine.py` / `dj_analysis.py` / etc.) — per your own
`CLAUDE.md`, never fold fixes into `classic_engine/`, which is a frozen, byte-faithful vendored
snapshot.

### 5.1 Confidence-gated fallback (instead of always forcing a full transition)

**Precedent:** Apple ships AutoMix *and* Crossfade side by side rather than forcing one aggressive mode
on every pair of tracks; Spotify's own docs concede its Mix tool "works best" only on compatible tracks
rather than claiming to fix incompatible ones. Your engine currently always commits to the full
stem-aware pipeline regardless of how compatible the two tracks actually are.

```text
Goal: Add a transition-confidence gate to the current-engine remix path. Before committing to the
full stem-aware crossfade, compute a composite confidence score from signals already available
(tempo-stretch ratio magnitude, TIV harmonic score, spectral_matching's best-candidate score) and
fall back to a simpler, safe transition when confidence is below a threshold, instead of forcing a
bad blend.

Constraints:
- current-engine only. Do not touch scripts/core/classic_engine/ — it's a frozen vendored snapshot
  per CLAUDE.md and must stay byte-faithful.
- New config keys go in config.yaml AND need a matching dataclass field in scripts/core/config.py's
  _dict_to_config() — CLAUDE.md flags this as a silent-failure trap, guarded by tests/test_config_drift.py.
- Don't do async I/O inside the task function — job_store.submit_job() runs it in a ThreadPoolExecutor.

Files likely touched:
- scripts/core/dj_engine.py (fallback render path — probably a straight equal-power crossfade at
  reduced duration, no stem split)
- scripts/api/task_modules/remix.py (where the confidence score gets computed and the branch decision
  gets made, alongside where spectral_matching already adjusts plan.entry_bar_b)
- config.yaml (new dj.transition_confidence_threshold key + matching AppConfig field)
- tests/test_behavioral.py (add a case: force a low-confidence pair, assert the fallback path renders
  instead of the full pipeline)

Definition of done: a deliberately incompatible track pair (large tempo gap, low TIV score) renders
via the fallback path and the job result's metadata records which path was used; a compatible pair
still gets the full pipeline; existing tests still pass.
```

### 5.2 Segment-level key confidence, not one whole-track guess

**Precedent:** Spotify's (now-dead) Echo Nest API scored *segments*, not whole tracks — each with its
own confidence. Your `key_detection.py` currently gives one Camelot classification per song, which is
wrong for anything that modulates key mid-track, and gives no signal for when to distrust the result.

```text
Goal: Add confidence-scored, per-phrase key estimation instead of a single whole-track key, so
downstream harmonic scoring (Camelot + TIV) can flag when a track's key call is unreliable rather
than silently trusting it.

Constraints:
- CAMELOT and NOTE_NAMES in key_detection.py are the single source of truth per CLAUDE.md — don't
  add a second inline key/semitone mapping, it will drift from pitch_shift_for_camelot().
- This is analysis, not rendering — safe to touch outside classic_engine entirely; classic_engine
  has its own frozen analysis and isn't affected.

Files likely touched:
- scripts/core/key_detection.py (add a windowed Krumhansl-Schmuckler pass across phrase-length
  windows, derive a confidence score from cross-window agreement)
- scripts/core/dj_analysis.py (surface per-phrase key + confidence on SongStructure)
- scripts/core/tiv_scoring.py (gate/weight the harmonic score by the confidence signal instead of
  assuming the key call is always trustworthy)

Definition of done: a track with a known mid-song key change (test fixture or synthetic pitch-shifted
splice) produces two different phrase-level key calls with a visible confidence drop at the boundary,
instead of one averaged/wrong whole-track key.
```

### 5.3 Fast, preview-tier stem separation path

**Precedent:** Algoriddim/AudioShake's Neural Mix runs real-time source separation on mobile hardware.
Your own `CLAUDE.md` gotchas admit Demucs runs 2–5× real time on CPU — fine for a final render job,
bad for anything a user is watching load (PiP widget, MixDeck live preview).

```text
Goal: Add a lighter/faster separation tier for interactive contexts (live preview, PiP widget) that
trades stem quality for speed, reserving full 4-stem Demucs for final render jobs only.

Constraints:
- Don't change the default full-quality path or its output format — final renders must stay
  byte-comparable to what they are today for existing tests.
- New separation.* config keys need matching dataclass fields, same drift trap as above.
- Benchmark before committing to an approach — this is a "measure, don't guess" task per your own
  agentic habits; profile actual latency on this machine's device (config separation.device) before
  picking a model.

Files likely touched:
- scripts/core/stems.py (add a fast-path separation function — candidates to benchmark: a 2-stem
  drums+bass/vocals+other pass instead of full 4-stem, or a smaller/quantized Demucs variant)
- config.yaml (separation.preview_mode or similar)
- frontend calls that currently wait on full separation for a live preview (check lib/api.ts and the
  MixDeck/PiP widget flow)

Definition of done: a benchmark script (throwaway, in scratch, not committed) showing wall-clock time
for full vs. fast-path separation on 3 representative songs, plus the fast path wired into whichever
preview flow currently blocks on full separation.
```

### 5.4 Learned differentiable transition controller as a second tier

**Precedent:** DJTransGAN (ICASSP 2022) trains a GAN generator that IS a differentiable EQ+fader
controller, fit to 284 real DJ mixes. Your `spectral_matching.py` already does online per-band weight
correction from 👍/👎 feedback — this is the natural next tier, not a replacement.

```text
Goal: Prototype a small differentiable EQ+fader controller (in the spirit of DJTransGAN, arXiv:2110.06525)
trained on a corpus of transitions, as an alternative candidate generator that spectral_matching.py can
compare against its current hand-tuned cosine/equal-power curves — not a swap-in replacement.

Constraints:
- This is the most ambitious item here — scope it as a research prototype first (a script in
  ai_remixmate_feature_lab/, which is explicitly isolated and not wired into the UI by design per
  CLAUDE.md), not a production change to spectral_matching.py on day one.
- Needs a training corpus — start small: your own library's already-rendered transitions plus
  their 👍/👎 verdicts (data/spectral_match_history.json) is a legitimate bootstrap dataset before
  reaching for external DJ-mix scraping.

Files likely touched:
- ai_remixmate_feature_lab/ (new prototype module — keep it isolated per the project's existing
  design intent)
- Read-only reference to data/spectral_match_model.json / spectral_match_history.json for the
  bootstrap dataset
- Only once validated: a hook point in scripts/core/spectral_matching.py where the learned candidate
  gets compared against the current heuristic, gated behind a config flag, current-engine only

Definition of done: a notebook/script in the feature lab that trains the controller on your existing
history data and reports whether its predicted transition parameters correlate with 👍 verdicts better
than the current hand-tuned curves — a research result, not a merged feature, at this stage.
```

### 5.5 Extend the feedback loop from transition point to next-track selection

**Precedent:** Spotify's actual product strength is personalization, not DSP — its Discover
Weekly/DJ selection ML is the one thing in this whole document that's genuinely more advanced than
your engine. Your own `SELF_IMPROVING_DJ_RESEARCH.md` already frames the right pattern ("a small
online correction to a hand-tuned prior") — it's just scoped to transition point today, not to which
track plays next.

```text
Goal: Extend the existing 👍/👎 online-correction pattern (currently scoped to spectral_matching's
transition-point selection) one layer up into setlist_planner.py's greedy next-track cost function,
as a small learned correction on top of the hand-tuned Camelot/BPM/energy weights — not a replacement
for them.

Constraints:
- Fix the known bug first: REMIX_QUALITY_INSIGHTS.md finding #1 says the hand-tuned formula doesn't
  match its own documented weights — a learned correction on top of a broken prior corrects noise,
  not signal. Confirm/fix that before adding this.
- Keep it a *small* online correction, same philosophy as the existing spectral-match model — this
  is explicitly not "build a recommender system."

Files likely touched:
- scripts/core/setlist_planner.py (cost function gets a small learned bias term)
- New feedback capture point — likely a router alongside POST /dj-remix/feedback, scoped to
  setlist/next-track choices rather than transition points
- A new small JSON store (data/setlist_feedback_model.json), same pattern as
  data/spectral_match_model.json

Definition of done: skipping/thumbs-downing a specific track-to-track jump measurably lowers that
jump's score on repeat planning runs for that user/library, without breaking the existing Camelot/BPM/
energy-arc cost ordering for everything else.
```

### 5.6 Publish a stable, versioned audio-analysis contract

**Precedent:** Spotify's Echo Nest API was, for a decade, the most detailed public MIR contract any
streamer shipped — bars/beats/tatums/segments with confidence — and it just got killed for new
integrations (Nov 2024). That's a real gap in the public ecosystem, and you already compute the
equivalent internally in `dj_analysis.py`.

```text
Goal: Expose a versioned GET endpoint that returns the Echo-Nest-shaped structure (bars, beats,
sections, segments with pitch/timbre/confidence) your dj_analysis.py already computes internally,
as a stable public contract — useful for the frontend visualizer/MixDeck manual-override UI (the
same BPM/key/waveform-at-a-glance idea Spotify's Mix tool uses) and as a portfolio-visible
differentiator.

Constraints:
- Router registration order matters — analysis is deliberately registered before library in
  routes.py so library's dynamic GET /library/{name} doesn't shadow analysis's static routes.
  A new static route under analysis should be safe, but add a case to tests/test_route_shadowing.py
  to prove it.
- This is a public contract, not an internal debug endpoint — version it (e.g. under
  /analysis/v1/...) so internal changes to dj_analysis.py's data model don't silently break it later.

Files likely touched:
- scripts/api/routers/analysis.py (new GET /analysis/{name}/structure or similar)
- scripts/api/schemas.py (Pydantic response model mirroring the bars/beats/tatums/segments shape)
- frontend/src/lib/api.ts (client call)
- tests/test_route_shadowing.py (new case for the static route)

Definition of done: hitting the new endpoint for a library track returns a documented, versioned JSON
shape with bars/beats/segments/pitch/timbre/confidence; a short note in ARCHITECTURE.md or the API
docs pointing out this is deliberately filling the gap Spotify's deprecated API left.
```

---

## References

- [MusicRadar — "Sack The DJ: Apple Music now uses AI to mix tracks"](https://www.musicradar.com/music-tech/sack-the-dj-apple-music-now-uses-ai-to-mix-tracks)
- [MusicTech — Apple Music's AutoMix uses AI to time-stretch and beat-match](https://musictech.com/news/gear/apple-music-automix-ai/)
- [Mixmag — Apple Music launches AutoMix](https://mixmag.net/read/apple-music-launches-new-automix-feature-that-blends-tracks-together-using-ai-tech)
- [Apple Newsroom — Apple services deliver powerful features and intelligent updates this fall (June 10, 2025)](https://www.apple.com/newsroom/2025/06/apple-services-deliver-powerful-features-and-intelligent-updates-to-users-this-fall/)
- [MacRumors — iOS 26: Enable the New AutoMix Feature in Apple Music](https://www.macrumors.com/how-to/ios-enable-automix-feature-apple-music/)
- [MacRumors — Apple Music Gains AutoMix Upgrades and More in iOS 27 (June 9, 2026)](https://www.macrumors.com/2026/06/09/apple-music-gains-automix-upgrades-and-more-in-ios-27/)
- [MacRumors — Spotify Debuts Playlist Mixing Tool Ahead of Apple Music AutoMix (Aug 20, 2025)](https://www.macrumors.com/2025/08/20/spotify-debuts-playlist-mixing-tool/)
- [DJ Mag — You can now DJ with Apple Music using rekordbox, Serato and more](https://djmag.com/tech/you-can-now-dj-apple-music-using-rekordbox-serato-and-more)
- [Digital DJ Tips — Apple Music Comes To Serato, Rekordbox, EngineDJ...](https://www.digitaldjtips.com/apple-music-serato-rekordbox-alphatheta-numark-denondj-enginedj/)
- [Spotify Newsroom — Spotify Debuts a New AI DJ (Feb 22, 2023)](https://newsroom.spotify.com/2023-02-22/spotify-debuts-a-new-ai-dj-right-in-your-pocket/)
- [Popular Science — How Spotify's AI DJ works](https://www.popsci.com/technology/spotify-ai-dj/)
- [Variety — Spotify DJ: AI-Powered DJ Song Feature Uses OpenAI Technology](https://variety.com/2023/digital/news/spotify-dj-persaonlized-ai-openai-1235532195/)
- [Music Machinery — Syncing Echo Nest analysis to Spotify Playback](https://musicmachinery.com/2012/04/09/syncing-echo-nest-analysis-to-spotify-playback/)
- [Spotify for Developers — Introducing some changes to our Web API (Nov 27, 2024)](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)
- [TechCrunch — Spotify cuts developer access to several of its recommendation features](https://techcrunch.com/2024/11/27/spotify-cuts-developer-access-to-several-of-its-recommendation-features/)
- [Algoriddim — Neural Mix Pro](https://www.algoriddim.com/neural-mix)
- [DJTransGAN paper — arXiv:2110.06525](https://arxiv.org/abs/2110.06525)
- [FxNorm-Automix — Sony CSL project page](https://marco-martinez-sony.github.io/FxNorm-automix/)
- [Essentia — ISMIR audio analysis library](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/bogdanov_essentia_ismir13.pdf)
