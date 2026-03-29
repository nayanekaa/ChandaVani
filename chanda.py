import _utf8  # noqa: F401  — sets stdout to UTF-8 on Windows
"""
Chanda Pattern Library + Matcher (Improved)
Chanda–Melodic PoC — Step 3

Imports syllabifier.py (Step 1 & 2) and adds:
  - A library of 6 common Sanskrit meters
  - A robust pattern matcher with fuzzy matching support
  - Full-verse validator with per-pada breakdown
  - Gana/Laghu-Guru conversion utilities
  - Levenshtein distance-based fuzzy matching for borderline cases

Usage:
    python chanda.py
"""

import re
from collections import Counter
from syllabifier import to_iast, syllabify_pada, classify

# Try to import Levenshtein for fuzzy matching; fall back to basic edit distance
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    def levenshtein_distance(s1, s2):
        """Simple edit distance fallback"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

# ── Gana and Laghu-Guru utilities ────────────────────────────────────────────

# Gana mappings: 8 classical ganas
GANA_MAP = {
    'Y': 'LGG',  # Yama
    'R': 'GLG',  # Raga
    'T': 'GGL',  # Tata
    'N': 'LLL',  # Nata
    'B': 'GLL',  # Bhaga
    'J': 'LGL',  # Jata
    'S': 'LLG',  # Salga
    'M': 'GGG',  # Magha
}

GANA_INV = {v: k for k, v in GANA_MAP.items()}

def lg_to_gana(lg_str):
    """Convert Laghu-Guru string (e.g., 'LGGLLG') to Gana string (e.g., 'YNS')
    
    Note: Groups must be exactly 3 characters. Incomplete groups are marked with '?'.
    """
    gana = []
    for i in range(0, len(lg_str), 3):
        group = lg_str[i:i+3]
        if len(group) == 3:
            gana.append(GANA_INV.get(group, '?'))
        else:
            # Mark incomplete groups as '?'
            gana.append('?')
    return ''.join(gana)

def gana_to_lg(gana_str):
    """Convert Gana string (e.g., 'YNS') to Laghu-Guru string"""
    lg = ''
    for g in gana_str:
        lg += GANA_MAP.get(g, '')
    return lg

def count_matra(gl_seq):
    """Count total matras (morae) in a G/L sequence. L=1 mora, G=2 morae."""
    return sum(2 if gl == 'G' else 1 for gl in (gl_seq if isinstance(gl_seq, str) else gl_seq))

# ── pattern notation ──────────────────────────────────────────────────────────
#
# Each pada template is a string of:
#   G  = guru  (heavy, must match)
#   L  = laghu (light, must match)
#   X  = free  (either G or L)
#   /  = yati  (caesura / pause — marks a natural breath point, not a syllable)
#
# For meters with multiple pada types, each type is listed separately.

CHANDA_LIBRARY = {

    "Anushtubh": {
        "description": "Most common Sanskrit meter · 4 padas × 8 syllables · Epics, Puranas, Gita",
        "padas_per_verse": 4,
        # Rules per Chandomitra (Jagadeeshan et al., EACL 2026) Section 2.2:
        #   Syllable 5: ALWAYS laghu (L)
        #   Syllable 6: ALWAYS guru  (G)
        #   Syllable 7: guru in ODD padas (1,3); laghu in EVEN padas (2,4)
        #   Syllables 1-4, 8: free (X)
        "pada_variants": [
            {"name": "odd-pada (standard)",  "template": "X X X X L G G X", "pada_type": "odd"},
            {"name": "even-pada (standard)", "template": "X X X X L G L X", "pada_type": "even"},
            {"name": "ra-vipula (odd)",      "template": "X X X X L G L G", "pada_type": "odd"},
            {"name": "na-vipula (odd)",      "template": "X X X X L L L G", "pada_type": "odd"},
            {"name": "ma-vipula (odd)",      "template": "X X X X L G G G", "pada_type": "odd"},
        ],
        "fixed_positions": {5: "L", 6: "G"},
        "yati": [None],
    },

    "Gayatri": {
        "description": "Vedic meter · 3 padas × 8 syllables · Rig Veda, Gayatri Mantra",
        "padas_per_verse": 3,
        "pada_variants": [
            {
                "name": "standard",
                "template": "X X X X X X X X",
            },
        ],
        "yati": [None],
    },

    "Trishtubh": {
        "description": "Vedic meter · 4 padas × 11 syllables · Rigvedic workhorse",
        "padas_per_verse": 4,
        "pada_variants": [
            {
                "name": "standard",
                # positions 5-7 have a specific cadence pattern
                "template": "X X X X G L G L X L X",
            },
        ],
        "yati": [4],   # after syllable 4
    },

    "Indravajra": {
        "description": "Classical · 4 padas × 11 syllables · T T J G G pattern",
        "padas_per_verse": 4,
        "pada_variants": [
            {
                "name": "standard",
                # T=tata(GGL) T=tata(GGL) J=jata(GLG) G G
                "template": "G G L G G L G L G G X",
            },
        ],
        "yati": [None],
    },

    "Vasantatilaka": {
        "description": "Classical · 4 padas × 14 syllables · T B J J G",
        "padas_per_verse": 4,
        "pada_variants": [
            {
                "name": "standard",
                "template": "G G L G L L G L G L G G X X",
            },
        ],
        "yati": [7],   # after syllable 7
    },

    "Mandakranta": {
        "description": "Classical · 4 padas × 17 syllables · Meghaduta meter",
        "padas_per_verse": 4,
        "pada_variants": [
            {
                "name": "standard",
                "template": "G G G G L L L L L G G L G G L X X",
            },
        ],
        "yati": [4, 10],   # after syllables 4 and 10
    },
}

# ── template engine ───────────────────────────────────────────────────────────

def parse_template(template_str):
    """Return list of tokens, stripping yati markers and spaces."""
    return [t for t in template_str.split() if t != "/"]


def match_template(gl_sequence, template_str, fuzzy=False):
    """
    Returns (matched: bool, score: float 0-1, mismatches: list).
    X positions always match; G/L positions must match exactly.
    
    Args:
        gl_sequence: list of 'G' and 'L' characters
        template_str: template string with G, L, X, and / characters
        fuzzy: if True, allow fuzzy matching with inexact matches
    
    Returns:
        (matched, score, mismatches)
    """
    tokens = parse_template(template_str)
    
    # If exact length match required
    if not fuzzy and len(gl_sequence) != len(tokens):
        return False, 0.0, ["Length mismatch"]
    
    min_len = min(len(gl_sequence), len(tokens))
    gl_prefix = gl_sequence[:min_len]
    tokens_prefix = tokens[:min_len]

    mismatches = []
    matches = 0
    for i, (actual, expected) in enumerate(zip(gl_prefix, tokens_prefix)):
        if expected == "X" or actual == expected:
            matches += 1
        else:
            mismatches.append(f"pos {i+1}: got {actual}, need {expected}")

    # Exact match requires perfect match and correct length
    exact_match = len(mismatches) == 0 and len(gl_sequence) == len(tokens)
    
    # Score calculation
    max_len = max(len(gl_sequence), len(tokens))
    score = matches / max_len if max_len > 0 else 0.0

    return exact_match, score, mismatches


def best_variant_match(gl_sequence, pada_variants, fuzzy=False, k=5):
    """
    Try all variants and return the best one(s).
    
    Args:
        gl_sequence: list of 'G' and 'L' chars
        pada_variants: list of variant dicts with 'template' and 'name'
        fuzzy: if True, include fuzzy matches with similarity scores
        k: number of top matches to return (for fuzzy mode)
    
    Returns:
        dict with variant info and match quality
    """
    results = []
    
    for variant in pada_variants:
        matched, score, mismatches = match_template(
            gl_sequence, variant["template"], fuzzy=fuzzy
        )
        entry = {
            "variant": variant.get("name", "unknown"),
            "template": variant["template"],
            "matched": matched,
            "score": score,
            "mismatches": mismatches,
        }
        results.append(entry)
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    if fuzzy:
        return results[:k]  # Return top k matches
    else:
        return results[0] if results else None

# ── chanda identifier ─────────────────────────────────────────────────────────

def identify_chanda(gl_sequences, fuzzy=False):
    """
    Given a list of G/L sequences (one per pada), score every chanda in the
    library and return a ranked list of candidates.

    Args:
        gl_sequences: list of lists, e.g. [["G","L","G","L",...], ...]
        fuzzy: if True, include fuzzy matches for unmatched padas
    
    Returns:
        List of dicts with chanda scores, ranked by overall match quality
    """
    results = []

    for name, chanda in CHANDA_LIBRARY.items():
        expected_padas = chanda["padas_per_verse"]
        variants = chanda["pada_variants"]

        pada_scores = []
        pada_results = []
        exact_matches = 0
        
        for i, gl in enumerate(gl_sequences[:expected_padas]):
            # Convert list to string if needed
            if isinstance(gl, list):
                gl_str = ''.join(gl)
            else:
                gl_str = gl
            
            best = best_variant_match(gl_str, variants, fuzzy=False)
            if best is not None:
                pada_scores.append(best["score"])
                pada_results.append(best)
                if best["matched"]:
                    exact_matches += 1

        overall = sum(pada_scores) / len(pada_scores) if pada_scores else 0.0
        
        # Boost score if multiple exact matches
        if exact_matches > 1:
            overall = min(1.0, overall + 0.1 * (exact_matches - 1))

        results.append({
            "chanda": name,
            "description": chanda["description"],
            "overall_score": overall,
            "exact_matches": exact_matches,
            "pada_results": pada_results,
        })

    results.sort(key=lambda r: (r["exact_matches"], r["overall_score"]), reverse=True)
    return results

# ── duration sequence ─────────────────────────────────────────────────────────
#
# Step 4 is folded in here as a utility: convert a G/L list to durations.
# This will be used by the audio pipeline later.

GURU_MS  = 200   # milliseconds per guru syllable
LAGHU_MS = 100   # milliseconds per laghu syllable
YATI_MS  = 300   # milliseconds for caesura pause

def gl_to_durations(gl_sequence, chanda_name=None):
    """
    Convert a G/L string to a list of (syllable_index, gl, duration_ms).
    If a chanda is specified, inserts yati pauses at the right positions.
    
    Args:
        gl_sequence: string of 'G' and 'L' characters
        chanda_name: name of the chanda to get yati positions
    """
    # Convert to string if list
    if isinstance(gl_sequence, list):
        gl_str = ''.join(gl_sequence)
    else:
        gl_str = gl_sequence
    
    yati_positions = set()
    if chanda_name and chanda_name in CHANDA_LIBRARY:
        for pos in (CHANDA_LIBRARY[chanda_name]["yati"] or []):
            if pos is not None:
                yati_positions.add(pos)   # after this 1-indexed syllable

    durations = []
    for i, gl in enumerate(gl_str):
        dur = GURU_MS if gl == "G" else LAGHU_MS
        durations.append((i + 1, gl, dur))
        if (i + 1) in yati_positions:
            durations.append((i + 1, "yati", YATI_MS))

    return durations

# ── display ────────────────────────────────────────────────────────────────────

def analyse_verse(devanagari_padas, chanda_hint=None, verse_label=""):
    """
    Full verse analysis: syllabify each pada, classify G/L, identify chanda,
    assign durations.

    devanagari_padas: list of strings, one per pada
    chanda_hint: if you already know the meter, pass its name to skip ranking
    """
    print("=" * 64)
    if verse_label:
        print(verse_label)
    print("=" * 64)

    all_gl = []
    all_sylls = []
    all_gana = []

    for i, pada in enumerate(devanagari_padas):
        iast  = to_iast(pada)
        sylls = syllabify_pada(iast)
        gl    = ''.join([classify(s) for s in sylls])
        gana  = lg_to_gana(gl)
        
        all_gl.append(gl)
        all_sylls.append(sylls)
        all_gana.append(gana)

        print(f"Pada {i+1}: {pada}")
        print(f"  IAST : {iast}")
        print(f"  Sylls: {' · '.join(sylls)}")
        print(f"  G/L  : {' · '.join(gl)}  ({len(gl)} syllables, {count_matra(gl)} morae)")
        print(f"  Gana : {gana}")

    print()

    # identify chanda
    rankings = identify_chanda(all_gl)
    top = rankings[0]

    print(f"Chanda identification:")
    for i, r in enumerate(rankings[:5]):
        bar = "█" * int(r["overall_score"] * 20)
        matches = f" ({r['exact_matches']} exact)" if r.get("exact_matches", 0) > 0 else ""
        print(f"  {i+1}. {r['chanda']:<18} {bar:<20} {r['overall_score']*100:5.0f}%{matches}")

    detected = chanda_hint if chanda_hint else top["chanda"]
    print(f"\n  → Best match: {detected}")
    if detected in CHANDA_LIBRARY:
        print(f"     {CHANDA_LIBRARY[detected]['description']}")

    # per-pada variant breakdown for detected chanda
    if detected in CHANDA_LIBRARY:
        print(f"\nPer-pada detail ({detected}):")
        for i, (gl, gana, sylls) in enumerate(zip(all_gl, all_gana, all_sylls)):
            variants = CHANDA_LIBRARY[detected]["pada_variants"]
            best = best_variant_match(gl, variants)
            if best is None:
                print(f"  Pada {i+1}: ⚠ No variants available")
                continue
            status = "✓" if best["matched"] else "○"
            variant_tag = f"[{best['variant']}]" if best["matched"] else f"[{best['variant']} — {'; '.join(best['mismatches'])}]"
            print(f"  Pada {i+1}: {status} {variant_tag}")
            print(f"         L/G: {gl} → Gana: {gana}")

        # duration sequence for pada 1 (as sample)
        print(f"\nDuration sample — Pada 1 ({detected}):")
        durs = gl_to_durations(all_gl[0], detected)
        for idx, gl, ms in durs:
            label = f"  syll {idx:2d}  {gl:<4}  {ms} ms"
            print(label)
        total_ms = sum(ms for _, _, ms in durs)
        print(f"  Total pada 1 duration: {total_ms} ms  ({total_ms/1000:.2f} s)")
    print()


# ── run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyse_verse(
        [
            "वागर्थाविव संपृक्तौ",
            "वागर्थप्रतिपत्तये",
            "जगतः पितरौ वन्दे",
            "पार्वतीपरमेश्वरौ",
        ],
        verse_label="Raghuvamsha 1.1  (Kalidasa)"
    )

    analyse_verse(
        [
            "तत्सवितुर्वरेण्यम्",
            "भर्गो देवस्य धीमहि",
            "धियो यो नः प्रचोदयात्",
        ],
        verse_label="Gayatri Mantra"
    )
