import _utf8  # noqa: F401
"""
Melodic Model — Chanda–Melodic PoC
====================================

Research basis:
  Rama & Lakshmanan (2010), Section 6 — "The Musical Component of the
  Speech Synthesizer":

    "Indian music ... has 12 notes in each octave ... sa ri(soft) ri
     ga(soft) ga ma ma(sharp) pa dha(soft) dha ni(soft) ni.
     The note pa is the middle note (value 0). Slight changes to the
     frequency of the recorded audio unit file results in a change of
     the musical note."

  Their Table 2 values (semitone offsets from pa=0):
    sa=-7  ri_soft=-6  ri=-5  ga_soft=-4  ga=-3  ma=-2  ma_sharp=-1
    pa=0   dha_soft=1  dha=2  ni_soft=3   ni=4

Five selectable frameworks:
  1. vedic_svara    — 3-level Vedic pitch accent (Anudatta / Svarita / Udatta)
  2. pitch_contour  — smooth parabolic arc per pada (rise → peak → fall)
  3. raga_bhairav   — Raga Bhairav (morning, devotional)
  4. raga_yaman     — Raga Yaman (evening, expansive)
  5. paper2_tables  — directly uses Paper 2's empirical pitch arrays
                      (best match to actual Vedic recitation tradition)
"""

import math

# ── Base reference pitch ──────────────────────────────────────────────────────
# Paper 2 uses pa (Pa, 5th note of scale) as the middle reference (value=0).
# We set pa = D4 = 294 Hz — a comfortable male speaking/chanting fundamental.
# Female voices are shifted +5 semitones in audio.py before synthesis.

PA_HZ    = 294.0   # D4 — middle reference note
BASE_HZ  = PA_HZ   # alias kept for backward compat

def semitones_to_hz(n: float, base: float = PA_HZ) -> float:
    return base * (2 ** (n / 12.0))

# ── Paper 2: 12-note system ───────────────────────────────────────────────────
# Table 2 from Rama & Lakshmanan (2010)
NOTE_OFFSET = {
    "sa":       -7,
    "ri_soft":  -6,
    "ri":       -5,
    "ga_soft":  -4,
    "ga":       -3,
    "ma":       -2,
    "ma_sharp": -1,
    "pa":        0,
    "dha_soft":  1,
    "dha":       2,
    "ni_soft":   3,
    "ni":        4,
}

NOTE_HZ = {k: round(semitones_to_hz(v), 2) for k, v in NOTE_OFFSET.items()}

# ── Raga definitions (aroha / avaroha + note names) ───────────────────────────

RAGAS = {
    "raga_bhairav": {
        "name": "Bhairav",
        "mood": "Meditative · devotional · dawn",
        # Sa Re♭ Ga Ma Pa Dha♭ Ni Sa  (komal Re and Dha)
        "aroha":   ["sa", "ri_soft", "ga", "ma", "pa", "dha_soft", "ni",     "sa"],
        "avaroha": ["sa", "ni",      "dha_soft", "pa", "ma", "ga",  "ri_soft", "sa"],
        "vadi":    "ga",       # prominent note
        "samvadi": "dha_soft", # second prominent
        "display": ["Sa", "Re♭", "Ga", "Ma", "Pa", "Dha♭", "Ni", "Sa'"],
    },
    "raga_yaman": {
        "name": "Yaman",
        "mood": "Expansive · devotional · evening",
        # Sa Re Ga Ma# Pa Dha Ni Sa  (teevra Ma)
        "aroha":   ["sa", "ri",  "ga", "ma_sharp", "pa", "dha", "ni",  "sa"],
        "avaroha": ["sa", "ni",  "dha", "pa", "ma_sharp", "ga",  "ri", "sa"],
        "vadi":    "ga",
        "samvadi": "ni",
        "display": ["Sa", "Re", "Ga", "Ma♯", "Pa", "Dha", "Ni", "Sa'"],
    },
    "raga_bhairavi": {
        "name": "Bhairavi",
        "mood": "Tender · emotional · morning",
        # Sa Re♭ Ga♭ Ma Pa Dha♭ Ni♭ Sa
        "aroha":   ["sa", "ri_soft",  "ga_soft", "ma", "pa", "dha_soft", "ni_soft", "sa"],
        "avaroha": ["sa", "ni_soft",  "dha_soft", "pa", "ma", "ga_soft", "ri_soft", "sa"],
        "vadi":    "ma",
        "samvadi": "sa",
        "display": ["Sa", "Re♭", "Ga♭", "Ma", "Pa", "Dha♭", "Ni♭", "Sa'"],
    },
}

def _raga_scale_hz(raga_key: str) -> list:
    """Return list of (note_name_display, hz) for all raga notes (unique)."""
    raga  = RAGAS[raga_key]
    seen  = set()
    scale = []
    for note in raga["aroha"] + raga["avaroha"]:
        if note not in seen:
            seen.add(note)
            scale.append(note)
    return scale  # list of note keys

def _nearest_note(target_semi: float, scale_keys: list) -> str:
    return min(scale_keys, key=lambda k: abs(NOTE_OFFSET[k] - target_semi))


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAMEWORK 1 — Vedic Svara
# ═══════════════════════════════════════════════════════════════════════════════

SVARA_HZ = {
    "anudatta": semitones_to_hz(-2),   # low
    "svarita":  semitones_to_hz( 0),   # mid (tonic pa)
    "udatta":   semitones_to_hz(+2),   # high
}

def assign_vedic_svara(syllables_gl: list) -> list:
    n = len(syllables_gl)
    out = []
    for i, (syll, gl) in enumerate(syllables_gl):
        pos = i + 1
        if pos == 1:
            svara = "svarita"
        elif pos == n:
            svara = "anudatta"
        elif pos >= n - 2:
            svara = "udatta"
        elif pos == n - 3:
            svara = "anudatta"
        else:
            svara = "svarita"
        hz = SVARA_HZ[svara] * (2 ** (2/12) if gl == "G" else 1.0)
        out.append((syll, gl, svara, round(hz, 1)))
    raw_hz = [hz for syll, gl, svara, hz in out]
    smoothed_hz = smooth_pitches(raw_hz)
    out = [(syll, gl, svara, round(h, 1)) for (syll, gl, svara, _), h in zip(out, smoothed_hz)]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAMEWORK 2 — Pitch Contour
# ═══════════════════════════════════════════════════════════════════════════════

def assign_pitch_contour(syllables_gl: list) -> list:
    n = len(syllables_gl)
    out = []
    for i, (syll, gl) in enumerate(syllables_gl):
        t      = i / max(n - 1, 1)
        peak_t = 0.6
        if t <= peak_t:
            st = 4.0 * (t / peak_t)
        else:
            st = 4.0 + (-2.0 - 4.0) * ((t - peak_t) / (1.0 - peak_t))
        if gl == "G":
            st += 2.0
        hz = semitones_to_hz(st)
        out.append((syll, gl, f"{hz:.0f}Hz", round(hz, 1)))
    raw_hz = [hz for syll, gl, lbl, hz in out]
    smoothed_hz = smooth_pitches(raw_hz)
    out = [(syll, gl, lbl, round(h, 1)) for (syll, gl, lbl, _), h in zip(out, smoothed_hz)]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAMEWORKS 3-5 — Raga-based
# ═══════════════════════════════════════════════════════════════════════════════

def assign_raga(syllables_gl: list, raga_key: str) -> list:
    """
    Assign raga-note pitches to syllables.

    Algorithm:
      - First half of pada → aroha (ascending); second → avaroha (descending)
      - Guru syllables → pulled toward vadi or samvadi
      - Uses Paper 2 note system for Hz values
    """
    raga      = RAGAS[raga_key]
    aroha     = raga["aroha"]
    avaroha   = raga["avaroha"]
    vadi      = raga["vadi"]
    samvadi   = raga["samvadi"]
    display   = raga["display"]
    scale_all = _raga_scale_hz(raga_key)

    n   = len(syllables_gl)
    out = []

    for i, (syll, gl) in enumerate(syllables_gl):
        t = i / max(n - 1, 1)

        # Select scale position
        if t < 0.5:
            scale  = aroha
            idx    = int(t * 2 * (len(scale) - 1))
        else:
            scale  = avaroha
            idx    = int((t - 0.5) * 2 * (len(scale) - 1))

        idx       = max(0, min(idx, len(scale) - 1))
        note_key  = scale[idx]

        # Guru: elevate by 2 semitones
        elevation = 2 if gl == "G" else 0
        hz        = round(NOTE_HZ[note_key] * (2 ** (elevation / 12)), 2)
        note_idx  = scale_all.index(note_key) if note_key in scale_all else 0
        note_name = display[min(note_idx, len(display)-1)]

        out.append((syll, gl, note_name, hz))
    raw_hz = [hz for syll, gl, note, hz in out]
    smoothed_hz = smooth_pitches(raw_hz)
    out = [(syll, gl, note, round(h, 2)) for (syll, gl, note, _), h in zip(out, smoothed_hz)]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAMEWORK 5 — Paper 2 Tables (best for hackathon — cites paper directly)
# ═══════════════════════════════════════════════════════════════════════════════
# Uses the same METER_PITCH_ARRAYS from audio.py.
# This framework label is used in the UI; actual array override happens
# in audio.py:apply_paper2_pitch_arrays() after assign_pitch() returns.
# Here we just assign a smooth contour as placeholder — the real pitch
# values are injected later in app.py.

def assign_paper2(syllables_gl: list) -> list:
    """
    Placeholder — returns contour Hz values.
    The actual Paper 2 pitch arrays are applied in audio.py after synthesis
    planning, overriding these values with the meter-specific note sequence.
    """
    return assign_pitch_contour(syllables_gl)


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified interface
# ═══════════════════════════════════════════════════════════════════════════════

FRAMEWORKS = ["vedic_svara", "pitch_contour", "raga_bhairav",
              "raga_yaman", "raga_bhairavi", "paper2_tables"]

FRAMEWORK_DESCRIPTIONS = {
    "vedic_svara":   "3-level Vedic pitch accent: Anudatta (low) / Svarita (mid) / Udatta (high)",
    "pitch_contour": "Smooth parabolic arc per pada: rises to peak at 60%, then descends",
    "raga_bhairav":  "Raga Bhairav — dawn, devotional, meditative (komal Re, komal Dha)",
    "raga_yaman":    "Raga Yaman — evening, expansive, devotional (teevra Ma)",
    "raga_bhairavi": "Raga Bhairavi — morning, tender, emotional (all komal notes)",
    "paper2_tables": "Paper 2 empirical pitch arrays (Rama & Lakshmanan 2010, Tables 3–4) — meter-specific note sequences from actual recitation analysis",
}


def assign_pitch(syllables_gl: list, framework: str = "vedic_svara") -> list:
    """
    syllables_gl : list of (syllable_str, gl_str)
    framework    : one of FRAMEWORKS

    Returns list of dicts: {syll, gl, hz, label}
    """
    if framework == "vedic_svara":
        raw = assign_vedic_svara(syllables_gl)
        return [{"syll": s, "gl": g, "label": sv,   "hz": hz} for s, g, sv, hz in raw]

    elif framework == "pitch_contour":
        raw = assign_pitch_contour(syllables_gl)
        return [{"syll": s, "gl": g, "label": lbl,  "hz": hz} for s, g, lbl, hz in raw]

    elif framework in ("raga_bhairav", "raga_yaman", "raga_bhairavi"):
        raw = assign_raga(syllables_gl, framework)
        return [{"syll": s, "gl": g, "label": note, "hz": hz} for s, g, note, hz in raw]

    elif framework == "paper2_tables":
        raw = assign_paper2(syllables_gl)
        return [{"syll": s, "gl": g, "label": lbl,  "hz": hz} for s, g, lbl, hz in raw]

    else:
        raise ValueError(f"Unknown framework: {framework!r}. "
                         f"Choose from: {FRAMEWORKS}")


def smooth_pitches(pitches_hz, alpha=0.3):
    """Apply exponential smoothing to reduce abrupt jumps."""
    if not pitches_hz:
        return pitches_hz
    smoothed = [pitches_hz[0]]
    for p in pitches_hz[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return smoothed

def analyse_melodic(devanagari_pada: str, framework: str = "vedic_svara",
                    chanda: str = "Anushtubh", label: str = ""):
    from syllabifier import to_iast, syllabify_pada, classify
    from chanda import gl_to_durations

    iast      = to_iast(devanagari_pada)
    sylls     = syllabify_pada(iast)
    gl        = [classify(s) for s in sylls]
    pitched   = assign_pitch(list(zip(sylls, gl)), framework)
    durations = gl_to_durations(gl, chanda)

    tag = f"[{label}]  " if label else ""
    print(f"\n{tag}{devanagari_pada}  [{framework}]")
    print(f"  {'Syllable':<10} {'G/L':<4} {'Pitch':<12} {'Hz':>7}  {'ms':>6}")
    print(f"  {'-'*8:<10} {'-'*3:<4} {'-'*10:<12} {'-'*6:>7}  {'-'*4:>6}")
    for i, p in enumerate(pitched):
        ms = durations[i][2] if i < len(durations) else "-"
        print(f"  {p['syll']:<10} {p['gl']:<4} {p['label']:<12} {p['hz']:>7.1f}  {ms:>6}")
    print(f"  Total: {sum(d[2] for d in durations)} ms")


if __name__ == "__main__":
    PADA = "वागर्थाविव संपृक्तौ"
    print("=" * 62)
    print("Raghuvamsha 1.1, Pada 1 — all frameworks")
    print("=" * 62)
    for fw in FRAMEWORKS:
        analyse_melodic(PADA, framework=fw, chanda="Anushtubh", label=fw)
