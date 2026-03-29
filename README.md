# 🕉 ChandaVani — Sanskrit Melodic Recitation Engine

> A research-backed pipeline that takes **raw Devanagari Sanskrit verse** and
> produces **metrically-correct, melodically-shaped audio** — enforcing
> Guru/Laghu rhythm, assigning Indian svara (Sa Re Ga Ma Pa) pitches,
> and mixing a tanpura drone bed. Every algorithmic step is grounded in
> peer-reviewed computational linguistics and musicology research.

---

## Pipeline Overview

```
Devanagari Text
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Step 1 · Transliteration          syllabifier.py   │
│  Devanagari → IAST (ISO 15919)                      │
│  Library: indic-transliteration                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 2 · Syllabification          syllabifier.py   │
│  IAST string → onset + nucleus + coda tuples        │
│  Rules: anusvara/visarga = closed syllable          │
│  Source: Rama & Lakshmanan (2010a) §3               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 3 · Guru / Laghu Classification syllabifier.py│
│  G (heavy): long vowel OR closed syllable           │
│  L (light): short open syllable                     │
│  Source: Rama & Lakshmanan (2010a) §2               │
│          Jagadeeshan et al. (2026) §2.2             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 4 · Chanda Identification    chanda.py        │
│  G/L string matched against 7 meter templates       │
│  Anushtubh positional rules (pos 5=L, 6=G)          │
│  ra-/na-/ma-vipula variant detection                 │
│  Fuzzy fallback via Levenshtein edit distance        │
│  Source: Jagadeeshan et al. (2026) §2.2, Table 2   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 5 · Pitch Assignment         melodic.py       │
│  5 selectable frameworks:                           │
│  · paper2_tables — empirical meter-specific arrays  │
│    from actual Vedic pandit recordings              │
│    Source: Rama & Lakshmanan (2010b) Tables 3–4    │
│  · vedic_svara   — Anudatta/Svarita/Udatta (3-tone)│
│    Source: Vedic Pratishakhya tradition             │
│  · raga_bhairav / raga_yaman / raga_bhairavi        │
│    12-note system, semitone offsets from Pa=0       │
│    Source: Rama & Lakshmanan (2010b) Table 2        │
│  Exponential pitch smoothing between adjacent notes │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 6 · Duration Mapping         chanda.py        │
│  G = 200 ms,  L = 100 ms  (2:1 ratio)              │
│  Yati (caesura) pauses inserted at meter positions  │
│  Source: Rama & Lakshmanan (2010a) §4               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 7 · Audio Synthesis          audio.py         │
│  Primary: gTTS (Google TTS, Hindi/Indian locale)    │
│  Gentle tempo stretch: 22% pull toward meter timing │
│  Melodic drone: tanpura-style note per syllable,    │
│    one octave below voice, mixed at 12%             │
│  Tonic Sa reference drone at 8%                     │
│  Scipy Butterworth high-pass (60 Hz rumble removal) │
│  Fallback: NumPy additive synthesis if gTTS absent  │
│  Source (drone concept): Rama & Lakshmanan (2010b)§6│
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 8 · Evaluation               evaluate.py      │
│  · Rhythm accuracy  (length + pattern + yati)       │
│  · Melody consistency (guru elevation, smoothness)  │
│  · Syllable precision (Devanagari + diacritic check)│
│  · Full/Partial Anushtubh compliance                │
│    Source: Jagadeeshan et al. (2026) Table 5        │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
              WAV audio + JSON analysis
```

---

## Quick Start (local, ~5 min)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install ffmpeg (required for gTTS MP3 decoding)

| Platform | Command |
|---|---|
| **Windows** | Download [ffmpeg-release-essentials.zip](https://www.gyan.dev/ffmpeg/builds/) → extract → add `bin/` to PATH |
| **macOS** | `brew install ffmpeg` |
| **Linux** | `sudo apt install ffmpeg` |

### 3. Start the server
```bash
python -m uvicorn app:app --reload --port 8000
```

### 4. Open in browser
```
http://localhost:8000
```

---

## Project Structure

```
chandavani/
├── app.py           # FastAPI — REST endpoints + request/response models
├── audio.py         # Audio synthesis — gTTS + tanpura drone + post-processing
├── chanda.py        # Meter library + G/L pattern matching + duration mapping
├── melodic.py       # Pitch frameworks — raga, Vedic svara, empirical tables
├── syllabifier.py   # Devanagari → IAST → syllables → Guru/Laghu classifier
├── evaluate.py      # Evaluation metrics (rhythm, melody, Anushtubh compliance)
├── _utf8.py         # Windows UTF-8 stdout fix
├── requirements.txt
├── Dockerfile       # Hugging Face Spaces / Docker deployment
└── static/
    └── index.html   # Single-page UI — shows Note Map (Sa Re Ga Ma Pa),
                     #   syllable strip, chanda detection, raga info
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/recite` | Full pipeline → WAV + analysis JSON |
| `POST` | `/analyse` | Metrics only (no audio) |
| `GET` | `/chandas` | All supported meters with templates |
| `GET` | `/ragas` | All ragas with aroha/avaroha |

### `/recite` request body
```json
{
  "padas":     ["धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"],
  "chanda":    "Anushtubh",
  "framework": "paper2_tables",
  "voice":     "male"
}
```

---

## Supported Meters

| Chanda | Syllables/pada | Padas | Canonical source |
|---|---|---|---|
| Anushtubh | 8 | 4 | Bhagavad Gita, Ramayana, Mahabharata |
| Gayatri | 8 | 3 | Rigveda — Gayatri Mantra (RV 3.62.10) |
| Trishtubh | 11 | 4 | Rigveda |
| Indravajra | 11 | 4 | Kalidasa — Kumarasambhava |
| Upendravajra | 11 | 4 | Classical Sanskrit |
| Vasantatilaka | 14 | 4 | Kalidasa — Meghaduta |
| Mandakranta | 17 | 4 | Kalidasa — Meghaduta |

Anushtubh mandatory constraints (pos 5 = Laghu, pos 6 = Guru) and
ra-/na-/ma-vipula variants implemented per
**Jagadeeshan et al. (2026), Section 2.2**.

---

## Melodic Frameworks & the Note System

All five frameworks use the same 12-note Indian octave anchored at
**Pa = 207 Hz** (G#3, calibrated from reference pandit recording).
Semitone offsets follow **Rama & Lakshmanan (2010b), Table 2**:

| Note | Offset from Pa | ~Hz |
|---|---|---|
| Sa | −7 | 130 |
| Ri♭ | −6 | 138 |
| Ri | −5 | 146 |
| Ga♭ | −4 | 155 |
| Ga | −3 | 164 |
| Ma | −2 | 174 |
| Ma♯ | −1 | 184 |
| **Pa** | **0** | **207** |
| Dha♭ | +1 | 219 |
| Dha | +2 | 232 |
| Ni♭ | +3 | 246 |
| Ni | +4 | 261 |

### Framework details

| Key | Name | Basis |
|---|---|---|
| `paper2_tables` | Vedic Pitch Arrays | Rama & Lakshmanan (2010b) Tables 3–4: per-meter empirical note sequences extracted from actual pandit recordings |
| `vedic_svara` | 3-level Vedic accent | Anudatta (low) / Svarita (mid) / Udatta (high) — Vedic Pratishakhya tradition |
| `raga_bhairav` | Raga Bhairav | Sa Re♭ Ga Ma Pa Dha♭ Ni — dawn, devotional |
| `raga_yaman` | Raga Yaman | Sa Re Ga Ma♯ Pa Dha Ni — evening, expansive |
| `raga_bhairavi` | Raga Bhairavi | Sa Re♭ Ga♭ Ma Pa Dha♭ Ni♭ — morning, tender |

---

## Research Citations

All core algorithms, data tables, and evaluation metrics are grounded in
the following peer-reviewed publications.

---

### [P1] Metrical Classification Algorithm
> **Rama, N. & Lakshmanan, M. (2010).**
> *A Computational Algorithm for Metrical Classification of Verse.*
> International Journal of Computer Science Issues (IJCSI), 7(4).

**Used in this project:**
- `syllabifier.py` — Guru/Laghu classification rules (§2)
- `syllabifier.py` — syllabification algorithm: onset + nucleus + coda (§3)
- `chanda.py` — duration model: G=200 ms, L=100 ms, 2:1 ratio (§4)
- `chanda.py` — yati (caesura) position table per meter

---

### [P2] Text-to-Tuneful Speech Synthesis
> **Rama, N. & Lakshmanan, M. (2010).**
> *Text-to-Tuneful Speech Synthesis of Sanskrit Verse.*
> International Journal of Computer Science and Network Security (IJCSNS), 10(7).

**Used in this project:**
- `melodic.py` — 12-note Indian music system, semitone offsets from Pa=0 (Table 2)
- `audio.py` — per-meter empirical pitch arrays for Anushtubh, Gayatri, Trishtubh, Indravajra, Vasantatilaka, Mandakranta (Tables 3–4)
- `audio.py` — tanpura-style melodic drone concept (§6: "The Musical Component")
- `audio.py` — Pa = reference pitch calibrated from pandit recording

---

### [P3] Chandomitra — EACL 2026
> **Jagadeeshan, A., Kunchukuttan, A., Anand Kumar, M., Shyam, R., & Murali Krishna, G. (2026).**
> *Chandomitra: Towards Generating Structured Sanskrit Poetry.*
> Proceedings of the 19th Conference of the European Chapter of the Association
> for Computational Linguistics (EACL 2026).

**Used in this project:**
- `chanda.py` — Anushtubh mandatory positional constraints: pos 5 = L, pos 6 = G (Section 2.2)
- `chanda.py` — ra-vipula, na-vipula, ma-vipula variant templates (Section 2.2)
- `evaluate.py` — Full Anushtubh % and Partial Anushtubh % evaluation metrics (Table 5)
- `chanda.py` — general meter template representation (Section 2)

---

### [P4] PINGALA — Prosody-Aware Decoding
> **Jagadeeshan, A., et al. (2026).**
> *PINGALA: Prosody-Aware Decoding for Sanskrit Poetry.*
> arXiv preprint.

**Used in this project:**
- Design motivation for chanda-constrained synthesis pipeline
- Prosody-aware decoding concepts informing the G/L → pitch mapping architecture

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Word-boundary sandhi only | Cross-word sandhi junctions may misclassify G/L |
| Fixed 2:1 G:L duration ratio | Actual pandit tempos are chanda-specific and variable |
| All-X templates (Gayatri) score 100% | Corpus frequency prior would improve disambiguation |
| Raga pitch snaps to discrete notes | Authentic performance uses continuous microtonal pitch |
| NumPy fallback = tonal synthesis | Not natural speech — requires gTTS + ffmpeg for voice quality |

---

## Deployment

### Hugging Face Spaces (Docker — recommended)
1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces) → **Docker** SDK
2. Push this repo — `Dockerfile` is pre-configured for port 7860

### Render (free tier)
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Environment:** `PYTHONIOENCODING=utf-8`

> **Note:** Free Render tier sleeps after inactivity — first request takes ~30s.

---

## License

MIT
