import _utf8  # noqa: F401  — sets stdout to UTF-8 on Windows
"""
Sanskrit Syllabifier + Guru/Laghu Classifier
Chanda–Melodic PoC — Step 1 & 2

Usage:
    python syllabifier.py

Paste any Devanagari pada into analyse_pada() at the bottom.
"""

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ── character sets ─────────────────────────────────────────────────────────────

LONG_VOWELS    = set("āīūeoēō")
SHORT_VOWELS   = set("aiu")
ALL_VOWELS     = set("aāiīuūeēoōṛṝḷ")
VOWEL_DIGRAPHS = {"ai", "au"}
CODA_MARKERS   = {"ṃ", "ḥ"}   # anusvara, visarga → always close syllable → guru

# ── helpers ───────────────────────────────────────────────────────────────────

def is_vowel(ch):
    return ch in ALL_VOWELS and ch not in CODA_MARKERS

def to_iast(devanagari_text):
    return transliterate(devanagari_text, sanscript.DEVANAGARI, sanscript.IAST)

# ── syllabifier ───────────────────────────────────────────────────────────────

def syllabify_word(iast_word):
    """
    Syllabify a single IAST word (no spaces).
    Rules:
      - Each syllable = onset consonants + vowel nucleus + optional coda
      - Coda: anusvara (ṃ) or visarga (ḥ) immediately after the vowel
      - Syllable boundary before each vowel (except diphthongs)
    Returns list of syllable strings.
    """
    chars = list(iast_word.lower())
    syllables = []
    i = 0

    while i < len(chars):
        # Start a new syllable
        onset = ""
        nucleus = ""
        coda = ""

        # Accumulate onset consonants
        while i < len(chars) and not is_vowel(chars[i]):
            onset += chars[i]
            i += 1

        # Vowel nucleus (check for digraphs)
        if i < len(chars) and is_vowel(chars[i]):
            if i + 1 < len(chars) and (chars[i] + chars[i + 1]) in VOWEL_DIGRAPHS:
                nucleus = chars[i] + chars[i + 1]
                i += 2
            else:
                nucleus = chars[i]
                i += 1

            # Check for coda
            if i < len(chars) and not is_vowel(chars[i]) and chars[i] not in CODA_MARKERS:
                # Look ahead to see if this consonant should be coda
                j = i + 1
                has_following_vowel = False
                while j < len(chars) and not is_vowel(chars[j]):
                    j += 1
                if j < len(chars) and is_vowel(chars[j]):
                    # There is a following vowel, so this consonant becomes coda
                    coda = chars[i]
                    i += 1

            # Optional anusvara/visarga coda
            if i < len(chars) and chars[i] in CODA_MARKERS:
                coda = chars[i]
                i += 1

        # Build syllable
        if nucleus:  # Only add if we found a nucleus
            syllable = onset + nucleus + coda
            syllables.append(syllable)

    return syllables


def syllabify_pada(iast_pada):
    """
    Syllabify a full pada (remove spaces for chanda analysis).
    """
    iast_pada = iast_pada.replace(' ', '')
    return syllabify_word(iast_pada)

# ── guru / laghu classifier ────────────────────────────────────────────────────

def classify(syllable):
    """
    G (guru / heavy) if:
      - vowel nucleus is long (ā, ī, ū, e, o, ai, au), OR
      - syllable ends in a consonant, anusvara, or visarga (closed syllable)
    L (laghu / light) otherwise.
    """
    # find the vowel nucleus (check digraphs first)
    nucleus = ""
    i = 0
    while i < len(syllable):
        if i + 1 < len(syllable) and syllable[i:i+2] in VOWEL_DIGRAPHS:
            nucleus = syllable[i:i+2]
            break
        elif is_vowel(syllable[i]):
            nucleus = syllable[i]
            break
        i += 1

    has_long_vowel = (nucleus in VOWEL_DIGRAPHS) or (nucleus in ALL_VOWELS and nucleus not in SHORT_VOWELS)

    last = syllable[-1]
    ends_closed = (not is_vowel(last)) or (last in CODA_MARKERS)

    return "G" if (has_long_vowel or ends_closed) else "L"

# ── chanda check ───────────────────────────────────────────────────────────────

CHANDA_RULES = {
    "Anushtubh / Shloka": {
        "syllables_per_pada": 8,
        # positions are 1-indexed; None means free
        "fixed": {5: "G", 6: "L", 7: "G"},
        "description": "8 syllables/pada · positions 5-6-7 must be G-L-G"
    },
    "Gayatri": {
        "syllables_per_pada": 8,
        "fixed": {},
        "description": "3 padas × 8 syllables · no fixed internal pattern"
    },
    "Trishtubh": {
        "syllables_per_pada": 11,
        "fixed": {},
        "description": "11 syllables/pada · Vedic; Rigvedic workhorse"
    },
}

def check_chanda(gl_sequence, chanda_name="Anushtubh / Shloka"):
    rule = CHANDA_RULES.get(chanda_name)
    if not rule:
        return f"Unknown chanda: {chanda_name}"

    issues = []
    expected = rule["syllables_per_pada"]
    actual = len(gl_sequence)

    if actual != expected:
        issues.append(f"count {actual} ≠ expected {expected}")

    for pos, expected_gl in rule["fixed"].items():
        if pos <= actual and gl_sequence[pos - 1] != expected_gl:
            issues.append(f"position {pos} is {gl_sequence[pos-1]}, expected {expected_gl}")

    if issues:
        return "⚠  " + " · ".join(issues)
    return "✓  matches " + chanda_name

# ── display ────────────────────────────────────────────────────────────────────

def analyse_pada(devanagari_pada, chanda="Anushtubh / Shloka", label=""):
    iast      = to_iast(devanagari_pada)
    sylls     = syllabify_pada(iast)
    gl        = [classify(s) for s in sylls]
    check     = check_chanda(gl, chanda)

    tag = f"[{label}] " if label else ""
    print(f"{tag}Devanagari : {devanagari_pada}")
    print(f"  IAST     : {iast}")
    print(f"  Syllables: {' · '.join(sylls)}")
    print(f"  G/L      : {' · '.join(gl)}")
    print(f"  Count    : {len(sylls)}")
    print(f"  Chanda   : {check}")
    print()

# ── test verses ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Raghuvamsha 1.1  (Anushtubh / Shloka)")
    print("=" * 60)
    analyse_pada("वागर्थाविव संपृक्तौ",        chanda="Anushtubh / Shloka", label="pada 1")
    analyse_pada("वागर्थप्रतिपत्तये",           chanda="Anushtubh / Shloka", label="pada 2")
    analyse_pada("जगतः पितरौ वन्दे",            chanda="Anushtubh / Shloka", label="pada 3")
    analyse_pada("पार्वतीपरमेश्वरौ",            chanda="Anushtubh / Shloka", label="pada 4")

    print("=" * 60)
    print("Gayatri Mantra  (Gayatri chandas — 3 padas, skip vyahrti)")
    print("=" * 60)
    analyse_pada("तत्सवितुर्वरेण्यम्",          chanda="Gayatri", label="pada 1")
    analyse_pada("भर्गो देवस्य धीमहि",          chanda="Gayatri", label="pada 2")
    analyse_pada("धियो यो नः प्रचोदयात्",       chanda="Gayatri", label="pada 3")
