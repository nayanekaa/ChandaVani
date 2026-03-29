import _utf8  # noqa
"""
ChandaVani — Audio Synthesis
==============================
Pipeline: gTTS Hindi → tempo blend → melodic drone mix → yati pauses → WAV

Melodic aspect:
  Every syllable generates a sustained tanpura-style drone note at its
  raga-assigned Hz (one octave below voice). Mixed at 22% volume under the
  voice — gives the chanting a musical, devotional quality without
  overpowering the words.

Backends: gtts_hindi (default, natural Indian voice) → numpy (always works)
"""

import io, os, wave, logging
import numpy as np
from scipy.signal import butter, sosfilt

log = logging.getLogger(__name__)

SAMPLE_RATE  = 22050
CHANNELS     = 1
PADA_GAP_MS  = 500
YATI_MS      = 280
CROSSFADE_MS = 55
DRONE_MIX    = 0.12   # melodic drone volume under voice

# ── Note table (Rama & Lakshmanan 2010, Table 2) ─────────────────────────────
PA_HZ = 207.0   # G#3 — calibrated from reference pandit recording

def note_val_to_hz(n): return PA_HZ * (2 ** (n / 12))

NOTE_HZ = {k: round(note_val_to_hz(v), 2) for k, v in {
    "sa":-7,"ri_soft":-6,"ri":-5,"ga_soft":-4,"ga":-3,
    "ma":-2,"ma_sharp":-1,"pa":0,"dha_soft":1,"dha":2,"ni_soft":3,"ni":4
}.items()}

METER_PITCH_ARRAYS = {
    "Anushtubh":     {"odd":[0,1,1,2,2,0,1,1],       "even":[0,1,-1,0,0,1,1,1]},
    "Gayatri":       {"odd":[-1,-1,-1,0,-1,-1,0,1],   "even":[-1,-1,-1,0,-1,-1,0,1]},
    "Indravajra":    {"odd":[0,0,1,2,2,0,0,1,-1,0,-1],"even":[0,1,0,0,0,0,-1,0,1,1,1]},
    "Upendravajra":  {"odd":[0,0,1,2,2,0,0,1,-1,0,-1],"even":[0,1,0,0,0,0,-1,0,1,1,1]},
    "Vasantatilaka": {"odd":[0,0,1,2,2,0,0,1,-1,0,-1,0,1,1],"even":[0,1,0,0,0,0,-1,0,1,1,1,0,0,1]},
    "Mandakranta":   {"odd":[0,0,0,0,-1,-1,-1,-1,-1,0,0,1,1,0,1,1,0],"even":[0,1,1,0,-1,-1,0,0,1,1,0,-1,-1,0,1,1,0]},
    "Trishtubh":     {"odd":[0,1,0,-1,0,1,2,1,0,1,0],"even":[0,1,0,-1,0,1,2,0,-1,0,1]},
    "_default":      {"odd":[0,1,1,2,2,0,1,1],       "even":[0,1,-1,0,0,1,1,1]},
}

def get_pitch_array(chanda, pada_index):
    arr = METER_PITCH_ARRAYS.get(chanda, METER_PITCH_ARRAYS["_default"])
    return arr["odd" if pada_index % 2 == 1 else "even"]

def apply_paper2_pitch_arrays(audio_padas, chanda):
    result = []
    for pi, pada_sylls in enumerate(audio_padas):
        pitch_arr = get_pitch_array(chanda, pi + 1)
        new = []
        for si, s in enumerate(pada_sylls):
            syl = dict(s)
            if si < len(pitch_arr):
                syl["hz"]       = round(note_val_to_hz(pitch_arr[si]), 2)
                syl["note_val"] = pitch_arr[si]
            new.append(syl)
        result.append(new)
    return result

VOICE_PROFILES = {"male": "Male (Pandit)", "female": "Female (Vedic)"}
BACKENDS       = ["gtts_hindi", "numpy"]


# ── Tempo stretch (scipy PSOLA fallback to linear) ────────────────────────────

def _tempo_stretch(audio: np.ndarray, target_ms: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    raw_ms = len(audio) / sr * 1000
    if raw_ms <= 0 or abs(raw_ms - target_ms) / max(raw_ms, 1) < 0.08:
        return audio
    try:
        import parselmouth
        from parselmouth.praat import call
        snd    = parselmouth.Sound(audio.astype(float), sr)
        dur    = snd.duration
        target = target_ms / 1000
        rate   = np.clip(dur / target, 0.45, 2.8)
        manip  = call(snd, "To Manipulation", 0.01, 50, 600)
        dt     = call(manip, "Extract duration tier")
        call(dt, "Remove points between", 0, dur)
        call(dt, "Add point", dur * 0.5, 1.0 / rate)
        call([dt, manip], "Replace duration tier")
        out = call(manip, "Get resynthesis (overlap-add)")
        return out.values.squeeze().astype(np.float32)
    except Exception:
        n_new = max(1, int(sr * target_ms / 1000))
        return np.interp(
            np.linspace(0, len(audio) - 1, n_new),
            np.arange(len(audio)), audio
        ).astype(np.float32)


# ── Yati pause insertion ───────────────────────────────────────────────────────

def _insert_yati_pauses(audio: np.ndarray, pada_sylls: list,
                        total_ms: int, pause_ms: int = YATI_MS) -> np.ndarray:
    yati_indices = [i for i, s in enumerate(pada_sylls) if s.get("yati_after")]
    if not yati_indices:
        return audio
    n_sylls = len(pada_sylls)
    ms_per = len(audio) / SAMPLE_RATE * 1000 / n_sylls if n_sylls else 0
    if ms_per <= 0:
        return audio
    parts = []
    silence = np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32)
    for i in range(n_sylls):
        s = int(i * ms_per / 1000 * SAMPLE_RATE)
        e = int((i + 1) * ms_per / 1000 * SAMPLE_RATE)
        parts.append(audio[s:e])
        if i in yati_indices:
            parts.append(silence)
    return np.concatenate(parts).astype(np.float32) if parts else audio


# ── Melodic drone (tanpura-style sustained notes) ────────────────────────────

def _drone_note(hz: float, dur_ms: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Tanpura-style sustained drone for one syllable.
    Generates the raga note one octave below voice — creates musical bed.
    """
    n   = max(128, int(sr * dur_ms / 1000))
    t   = np.linspace(0, dur_ms / 1000, n, dtype=np.float64)
    hz  = hz / 2   # one octave below voice
    # Slow vibrato (tanpura characteristic)
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * 3.5 * t)
    ph  = 2 * np.pi * hz * t * vib
    # Tanpura harmonics: strong fundamental, softer overtones
    w   = (np.sin(ph) + 0.35*np.sin(2*ph) + 0.12*np.sin(3*ph) +
           0.06*np.sin(4*ph)).astype(np.float32)
    pk  = np.max(np.abs(w))
    if pk > 0: w /= pk
    # Long attack, long release — sustained quality
    a, r = int(0.18*n), int(0.22*n)
    s    = max(0, n - a - r)
    env  = np.concatenate([np.linspace(0,1,a), np.full(s,0.85),
                           np.linspace(0.85,0,r)])
    env  = np.pad(env,(0,max(0,n-len(env))))[:n].astype(np.float32)
    return w * env


def _generate_drone_track(pada_sylls: list, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Build the full melodic drone bed for one pada."""
    parts = []
    for s in pada_sylls:
        parts.append(_drone_note(s["hz"], s["dur_ms"], sr))
        if s.get("yati_after"):
            parts.append(np.zeros(int(sr * YATI_MS / 1000), dtype=np.float32))
    if not parts:
        return np.zeros(sr, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def _add_tonic_tanpura(audio: np.ndarray, sr: int = SAMPLE_RATE,
                       level: float = 0.08) -> np.ndarray:
    """
    Add a very soft continuous Sa (tonic) drone throughout — traditional
    tanpura reference note that anchors all melodic movement.
    """
    n   = len(audio)
    t   = np.linspace(0, n / sr, n)
    sa  = PA_HZ / 4  # two octaves below voice, very low and warm
    vib = 1.0 + 0.002 * np.sin(2 * np.pi * 2.8 * t)
    w   = (np.sin(2*np.pi*sa*t*vib) + 0.25*np.sin(4*np.pi*sa*t)).astype(np.float32)
    pk  = np.max(np.abs(w)); w = w / pk if pk > 0 else w
    return audio + w * level


# ── Natural recitation blend ──────────────────────────────────────────────────

def _mix_voice_and_drone(voice: np.ndarray, drone: np.ndarray,
                          mix: float = DRONE_MIX) -> np.ndarray:
    """Mix TTS voice with melodic drone. Align lengths."""
    n = max(len(voice), len(drone))
    v = np.pad(voice, (0, n - len(voice))).astype(np.float32)
    d = np.pad(drone, (0, n - len(drone))).astype(np.float32)
    return v + d * mix


def _natural_recitation_pada(raw: np.ndarray, pada_sylls: list) -> np.ndarray:
    """
    Shape one pada:
      1. Gently stretch toward chanda timing (22% pull)
      2. Insert yati pauses
      3. Mix melodic drone underneath (22%)
      4. Add tonic tanpura reference (8%)
    """
    target_ms = sum(s["dur_ms"] for s in pada_sylls)
    raw_ms    = len(raw) / SAMPLE_RATE * 1000
    # 22% pull toward meter — preserves natural speech cadence
    blended   = int(round(0.78 * raw_ms + 0.22 * target_ms))
    blended   = int(np.clip(blended, raw_ms * 0.88, raw_ms * 1.15))
    paced     = _tempo_stretch(raw, blended)
    voiced    = _insert_yati_pauses(paced, pada_sylls, target_ms, YATI_MS)

    # Build drone matching the final audio length
    actual_ms = len(voiced) / SAMPLE_RATE * 1000
    ratio     = actual_ms / target_ms if target_ms > 0 else 1.0
    drone_sylls = [{**s, "dur_ms": int(s["dur_ms"] * ratio)} for s in pada_sylls]
    drone     = _generate_drone_track(drone_sylls)

    mixed     = _mix_voice_and_drone(voiced, drone, DRONE_MIX)
    mixed     = _add_tonic_tanpura(mixed)
    return mixed.astype(np.float32)


# ── Crossfade join ────────────────────────────────────────────────────────────

def _crossfade_join(segments: list, fade_ms: int = CROSSFADE_MS,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    if not segments:
        return np.zeros(sr, dtype=np.float32)
    out = segments[0].astype(np.float32)
    f   = int(sr * fade_ms / 1000)
    for nxt in segments[1:]:
        nxt = nxt.astype(np.float32)
        f   = min(f, len(out), len(nxt))
        if f > 2:
            out[-f:] *= np.linspace(1, 0, f)
            nxt[:f]  *= np.linspace(0, 1, f)
        out = np.concatenate([out, nxt])
    return out


# ── Post processing ───────────────────────────────────────────────────────────

def _post_process(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    # High-pass to remove sub-bass rumble
    sos = butter(4, 60, btype="high", fs=sr, output="sos")
    audio = sosfilt(sos, audio).astype(np.float32)
    # Normalise to -2 dB
    pk = np.max(np.abs(audio))
    if pk > 0:
        audio *= 0.794 / pk
    return audio


# ── gTTS Hindi backend ────────────────────────────────────────────────────────

def _gtts_ok():
    try:
        import gtts, pydub; return True  # noqa
    except ImportError:
        return False


def _arr_from_mp3(mp3_bytes: bytes) -> np.ndarray:
    from pydub import AudioSegment
    seg  = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    seg  = seg.set_channels(1).set_frame_rate(SAMPLE_RATE)
    arr  = np.array(seg.get_array_of_samples(), dtype=np.float32)
    arr /= 2 ** (seg.sample_width * 8 - 1)
    return arr


def _gtts_raw(text: str, female: bool = False) -> np.ndarray:
    from gtts import gTTS
    tts = gTTS(text=text, lang="hi", tld="co.in", slow=True)
    buf = io.BytesIO(); tts.write_to_fp(buf)
    return _arr_from_mp3(buf.getvalue())


# ── Numpy fallback ────────────────────────────────────────────────────────────

def _adsr(n, atk=0.06, dec=0.12, sus=0.72, rel=0.20):
    a = int(n*atk); d = int(n*dec); r = int(n*rel); s = max(0,n-a-d-r)
    e = np.concatenate([np.linspace(0,1,a),np.linspace(1,sus,d),
                        np.full(s,sus),np.linspace(sus,0,r)])
    return np.pad(e,(0,max(0,n-len(e))))[:n].astype(np.float32)


def _voiced_tone(hz, dur_ms, gl, sr=SAMPLE_RATE):
    n = max(256, int(sr*dur_ms/1000))
    t = np.linspace(0, dur_ms/1000, n)
    mod   = 1+(18/1200)*np.sin(2*np.pi*5.5*t) if gl=="G" else np.ones(n)
    phase = np.cumsum(2*np.pi*hz*mod/sr)
    w = (np.sin(phase)+0.55*np.sin(2*phase)+
         0.28*np.sin(3*phase)+0.14*np.sin(4*phase)).astype(np.float32)
    pk = np.max(np.abs(w)); w = w/pk if pk>0 else w
    return w * _adsr(n, atk=0.03 if gl=="G" else 0.012,
                         rel=0.08 if gl=="G" else 0.03)


def _numpy_pada(pada_sylls, sr=SAMPLE_RATE):
    f   = int(sr * 0.030)
    out = np.zeros(0, dtype=np.float32)
    sil = np.zeros(int(sr*YATI_MS/1000), dtype=np.float32)
    for s in pada_sylls:
        tone = _voiced_tone(s["hz"], s["dur_ms"], s["gl"], sr)
        if len(out) >= f > 0:
            out[-f:] *= np.linspace(1,0,f); tone[:f] *= np.linspace(0,1,f)
        out = np.concatenate([out, tone])
        if s.get("yati_after"): out = np.concatenate([out, sil])
    if len(out) == 0: return np.zeros(sr//2, dtype=np.float32)
    drone = _generate_drone_track(pada_sylls, sr)
    return _mix_voice_and_drone(out, drone, DRONE_MIX)


# ── WAV helper ────────────────────────────────────────────────────────────────

def _to_wav(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(audio,-1,1)*32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf,"wb") as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(2)
        wf.setframerate(sr);       wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── Synth one pada ────────────────────────────────────────────────────────────

def _synth_pada(deva_text, pada_sylls, voice):
    female = (voice == "female")
    raw    = None

    if _gtts_ok():
        try:
            raw = _gtts_raw(deva_text, female=female)
        except Exception as e:
            log.warning(f"gTTS failed ({e})")

    if raw is None or len(raw) < 64:
        log.warning("gTTS unavailable — numpy fallback")
        return _numpy_pada(pada_sylls).astype(np.float32)

    return _natural_recitation_pada(raw, pada_sylls)


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise_verse(
    audio_padas:         list,
    backend:             str  = "gtts_hindi",
    framework:           str  = "vedic_svara",
    voice:               str  = "male",
    original_deva_padas: list = None,
    pada_gap_ms:         int  = PADA_GAP_MS,
    **_kw,
) -> bytes:
    deva = original_deva_padas or ["" for _ in audio_padas]
    log.info(f"synthesise voice={voice!r} padas={len(audio_padas)}")

    pada_audios = []
    for pi, (sylls, deva_text) in enumerate(zip(audio_padas, deva)):
        if not deva_text.strip() or not sylls:
            continue
        try:
            audio = _synth_pada(deva_text, sylls, voice)
        except Exception as e:
            log.error(f"pada {pi+1} failed ({e})")
            audio = _numpy_pada(sylls)
        pada_audios.append(audio.astype(np.float32))

    if not pada_audios:
        return _to_wav(np.zeros(SAMPLE_RATE, dtype=np.float32))

    gap = np.zeros(int(SAMPLE_RATE * pada_gap_ms / 1000), dtype=np.float32)
    interleaved = []
    for i, pa in enumerate(pada_audios):
        interleaved.append(pa)
        if i < len(pada_audios) - 1:
            interleaved.append(gap.copy())

    combined = _crossfade_join(interleaved, fade_ms=CROSSFADE_MS)
    combined = _post_process(combined)
    return _to_wav(combined)
