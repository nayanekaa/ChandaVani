import _utf8  # noqa
"""
ChandaVani — FastAPI Backend

Run:
    pip install fastapi uvicorn[standard] gTTS pydub scipy numpy indic-transliteration soundfile python-multipart
    uvicorn app:app --reload --port 8000
    open http://localhost:8000
"""
import base64
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from syllabifier  import to_iast, syllabify_pada, classify
from chanda       import identify_chanda, gl_to_durations, CHANDA_LIBRARY, best_variant_match
from melodic      import assign_pitch, FRAMEWORKS
from audio        import synthesise_verse, apply_paper2_pitch_arrays
from evaluate     import evaluate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="ChandaVani")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ── Request / Response models ─────────────────────────────────────────────────

class ReciteRequest(BaseModel):
    padas:     List[str]
    chanda:    str  = "Anushtubh"
    framework: str  = "paper2_tables"
    voice:     str  = "male"          # "male" | "female"

class SyllableInfo(BaseModel):
    syll:    str
    gl:      str
    pitch:   str
    hz:      float
    dur_ms:  int

class PadaResult(BaseModel):
    pada:      str
    iast:      str
    syllables: List[SyllableInfo]
    variant:   str
    matched:   bool

class ReciteResponse(BaseModel):
    padas:         List[PadaResult]
    chanda_scores: dict
    audio_b64:     str
    total_ms:      int
    detected_chanda: str

class AnalyseRequest(BaseModel):
    padas:     List[str]
    chanda:    str = "Anushtubh"
    framework: str = "paper2_tables"

class MetricOut(BaseModel):
    name:      str
    score:     float
    label:     str
    rationale: str
    details:   dict

class AnalyseResponse(BaseModel):
    rhythm:               MetricOut
    melody:               MetricOut
    syllables:            MetricOut
    anushtubh_compliance: MetricOut
    overall:              float
    verdict:              str
    limitations:          List[str]
    suggestions:          List[str]


# ── /recite ───────────────────────────────────────────────────────────────────

@app.post("/recite", response_model=ReciteResponse)
async def recite(req: ReciteRequest):
    if req.chanda not in CHANDA_LIBRARY:
        raise HTTPException(400, f"Unknown chanda: {req.chanda}")
    if not req.padas:
        raise HTTPException(400, "Provide at least one pada")

    framework = req.framework if req.framework in FRAMEWORKS else "paper2_tables"

    pada_results, all_gl, audio_padas = [], [], []
    total_ms = 0

    for pada in req.padas:
        iast      = to_iast(pada)
        sylls     = syllabify_pada(iast)
        gl        = [classify(s) for s in sylls]
        all_gl.append(gl)

        pitched   = assign_pitch(list(zip(sylls, gl)), framework)
        durations = gl_to_durations(gl, req.chanda)
        best      = best_variant_match(gl, CHANDA_LIBRARY[req.chanda]["pada_variants"])
        yati_pos  = {p for p in (CHANDA_LIBRARY[req.chanda].get("yati") or []) if p}

        syll_infos, audio_sylls = [], []
        for i, p in enumerate(pitched):
            dur_ms     = durations[i][2] if i < len(durations) else 200
            yati_after = (i + 1) in yati_pos
            total_ms  += dur_ms
            syll_infos.append(SyllableInfo(
                syll=p["syll"], gl=p["gl"],
                pitch=p["label"], hz=p["hz"], dur_ms=dur_ms,
            ))
            audio_sylls.append({
                "syll": p["syll"], "gl": p["gl"],
                "hz": p["hz"], "dur_ms": dur_ms,
                "yati_after": yati_after,
            })

        pada_results.append(PadaResult(
            pada=pada, iast=iast, syllables=syll_infos,
            variant=best["variant"], matched=best["matched"],
        ))
        audio_padas.append(audio_sylls)

    all_chanda_results = identify_chanda(all_gl)
    chanda_scores = {
        r["chanda"]: round(r["overall_score"] * 100)
        for r in all_chanda_results[:5]
    }
    # Use the highest-scoring detected chanda for audio generation
    best_chanda = all_chanda_results[0]["chanda"] if all_chanda_results else req.chanda
    audio_padas = apply_paper2_pitch_arrays(audio_padas, best_chanda)

    wav = synthesise_verse(
        audio_padas, voice=req.voice,
        original_deva_padas=req.padas,
    )
    return ReciteResponse(
        padas=pada_results, chanda_scores=chanda_scores,
        audio_b64=base64.b64encode(wav).decode(), total_ms=total_ms,
        detected_chanda=best_chanda,
    )


# ── /analyse ──────────────────────────────────────────────────────────────────

@app.post("/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest):
    if req.chanda not in CHANDA_LIBRARY:
        raise HTTPException(400, f"Unknown chanda: {req.chanda}")
    framework = req.framework if req.framework in FRAMEWORKS else "paper2_tables"

    all_gl, pitched_padas = [], []
    for pada in req.padas:
        iast    = to_iast(pada)
        sylls   = syllabify_pada(iast)
        gl      = [classify(s) for s in sylls]
        pitched = assign_pitch(list(zip(sylls, gl)), framework)
        all_gl.append(gl); pitched_padas.append(pitched)

    report = evaluate(req.padas, req.chanda, framework, all_gl, pitched_padas)
    def to_out(m): return MetricOut(
        name=m.name, score=m.score, label=m.label,
        rationale=m.rationale, details=m.details,
    )
    return AnalyseResponse(
        rhythm=to_out(report.rhythm), melody=to_out(report.melody),
        syllables=to_out(report.syllables),
        anushtubh_compliance=to_out(report.anushtubh_compliance),
        overall=report.overall, verdict=report.verdict,
        limitations=report.limitations, suggestions=report.suggestions,
    )


# ── metadata ──────────────────────────────────────────────────────────────────

@app.get("/chandas")
async def list_chandas():
    return {
        name: {
            "description": info["description"],
            "syllables_per_pada": len(info["pada_variants"][0]["template"].split()) if info["pada_variants"] else 0,
        }
        for name, info in CHANDA_LIBRARY.items()
    }

@app.get("/ragas")
async def list_ragas():
    from melodic import RAGAS
    return {
        k: {"name": v["name"], "mood": v["mood"], "aroha": v["aroha"], "avaroha": v["avaroha"]}
        for k, v in RAGAS.items()
    }
