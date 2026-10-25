import re
import pandas as pd
from collections import Counter
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- Config ----------------

# Bulletproof NLTK download for Cloud deployment
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))

TECHNICAL_STOPWORDS = {
    "device", "method", "system", "apparatus", "means", "process",
    "assembly", "unit", "module", "structure", "article", "member",
    "portion", "element", "component", "arrangement", "mechanism",
    "related", "providing", "containing", "comprising", "including",
    "based", "thereof", "therein", "wherein", "thereby", "hereof",
    "said", "same", "such", "various", "plural", "predetermined",
    "first", "second", "third", "one", "two", "least",
}

STOPWORDS |= TECHNICAL_STOPWORDS

STEMMER = PorterStemmer()

SYNONYMS = {
    "artificial intelligence": "ai",
    "machine learning": "ml",
    "electric vehicle": "ev",
    "electric vehicles": "ev",
    "renewable energy": "renewable_energy",
    "solid state battery": "solid_state_battery",
    "solid-state battery": "solid_state_battery",
    "lithium ion battery": "li_ion_battery",
    "lithium-ion battery": "li_ion_battery",
}


# ---------------- 1. Load & clean data ----------------

def load_data(path):
    """Read a CSV and return a DataFrame with columns [Title, year]."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    title_col = _detect_column(df, kind="title")
    date_col = _detect_column(df, kind="date")

    if title_col is None:
        raise ValueError(
            f"No title column found. Columns in your file: {df.columns.tolist()}"
        )
    if date_col is None:
        raise ValueError(
            f"No year/date column found. Columns in your file: {df.columns.tolist()}"
        )

    # ----- Universal year extraction -----
    raw_dates = df[date_col].astype(str).str.strip()

    # Step 1: try plain numeric (handles "2019", "2010")
    df["year"] = pd.to_numeric(raw_dates, errors="coerce")

    # Step 2: fix YYYYMMDD integers (e.g. 20240229)
    mask_yyyymmdd = df["year"].notna() & (df["year"] > 3000)
    if mask_yyyymmdd.any():
        df.loc[mask_yyyymmdd, "year"] = pd.to_datetime(
            df.loc[mask_yyyymmdd, date_col].astype(str),
            format="%Y%m%d",
            errors="coerce",
        ).dt.year

    # Step 3: for anything still missing, parse as generic date string
    mask_missing = df["year"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "year"] = pd.to_datetime(
            df.loc[mask_missing, date_col],
            errors="coerce",
            dayfirst=True,   # handles DD/MM/YYYY (Indian / European)
            utc=True,        # handles "2021-04-15T00:00:00Z"
        ).dt.year

    # Step 4: drop rows with no valid year
    df = df.dropna(subset=[title_col, "year"])
    df["year"] = df["year"].astype(int)

    # Step 5: keep only reasonable years
    df = df[(df["year"] >= 1980) & (df["year"] <= 2030)]

    df = df.rename(columns={title_col: "Title"})
    return df[["Title", "year"]]


def _detect_column(df, kind):
    """Find a likely title or date column in a DataFrame."""
    if kind == "title":
        exact = [
            "Title", "title", "TITLE", "patent_title", "Patent Title",
            "invention_title", "Invention Title", "Title of invention",
            "Title of Invention", "title_of_invention", "Invention",
        ]
        fuzzy_keyword = "title"
        alt_keywords = ["invention", "subject", "name"]
    else:
        exact = [
            "Application Date", "application_date", "Application_Date",
            "year", "Year", "YEAR", "publication_date", "Publication Date",
            "Publication Date(U/S 11A)", "filing_date", "Filing Date",
            "Application Filing Date", "date", "Date", "Filed",
        ]
        fuzzy_keyword = "date"
        alt_keywords = ["year", "filed", "published"]

    for name in exact:
        if name in df.columns:
            return name

    for col in df.columns:
        low = col.lower()
        if fuzzy_keyword in low:
            return col

    for col in df.columns:
        low = col.lower()
        if any(k in low for k in alt_keywords):
            return col

    return None


# ---------------- 2. Clean text ----------------

@lru_cache(maxsize=100_000)
def _clean_and_stem(text):
    """Lowercase, strip, remove stopwords, and stem. Cached for speed."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    words = text.split()
    cleaned = []
    for w in words:
        if len(w) <= 2 or w in STOPWORDS:
            continue
        parts = w.replace("-", " ").split()
        for p in parts:
            if len(p) > 2 and p not in STOPWORDS:
                cleaned.append(STEMMER.stem(p))
    return cleaned


# ---------------- 3. Extract n-grams ----------------

def _extract_ngrams(titles, n=2, min_count=2):
    counter = Counter()
    for title in titles:
        words = _clean_and_stem(title)
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            counter[phrase] += 1

    filtered = {p: c for p, c in counter.items() if c >= min_count}
    return Counter(filtered)


def extract_bigrams(titles, min_count=2):
    return _extract_ngrams(titles, n=2, min_count=min_count)


def extract_trigrams(titles, min_count=2):
    return _extract_ngrams(titles, n=3, min_count=min_count)


# ---------------- 4. Compare years ----------------

def compare_years(df, base_year, current_year, min_count=2, use_trigrams=False):
    base_titles = df[df["year"] == base_year]["Title"]
    current_titles = df[df["year"] == current_year]["Title"]

    base_counts = extract_bigrams(base_titles, min_count=min_count)
    current_counts = extract_bigrams(current_titles, min_count=min_count)

    if use_trigrams:
        base_counts += extract_trigrams(base_titles, min_count=min_count)
        current_counts += extract_trigrams(current_titles, min_count=min_count)

    phrases = set(base_counts) | set(current_counts)

    rows = []
    for phrase in phrases:
        b = base_counts.get(phrase, 0)
        c = current_counts.get(phrase, 0)
        if b + c < min_count:
            continue
        canonical = SYNONYMS.get(phrase, phrase)
        rows.append({
            "phrase": phrase,
            "canonical": canonical,
            "base_count": b,
            "current_count": c,
        })

    return pd.DataFrame(rows)


# ---------------- 5. Score ----------------

def score_phrases(df, min_total=3):
    def growth(b, c):
        if b == 0:
            return 100.0 if c > 0 else 0.0
        return (c - b) / b * 100

    def bayes(b, c):
        return (1 + c) / (2 + b + c) * 100

    def confidence(b, c):
        total = b + c
        return min(100.0, total * 10)

    def trend(score, conf):
        if conf < 20:
            return "Low data"
        if score >= 60:
            return "Up"
        if score <= 40:
            return "Down"
        return "Neutral"

    df = df.copy()
    df["growth_ratio"] = df.apply(lambda r: growth(r.base_count, r.current_count), axis=1)
    df["score"] = df.apply(lambda r: bayes(r.base_count, r.current_count), axis=1)
    df["confidence"] = df.apply(lambda r: confidence(r.base_count, r.current_count), axis=1)
    df["trend"] = df.apply(lambda r: trend(r.score, r.confidence), axis=1)

    df = df[(df.base_count + df.current_count) >= min_total]

    return df.sort_values("score", ascending=False)


# ---------------- 6. Validation ----------------

def validate(df):
    """Return a list of warnings about the dataset."""
    warnings = []
    years = df["year"].unique()
    if len(years) < 2:
        warnings.append("Dataset contains only one year — trends cannot be computed.")
    if df["Title"].isna().mean() > 0.5:
        warnings.append("More than half of titles are missing.")
    if len(df) < 100:
        warnings.append(f"Only {len(df)} rows — results may be noisy.")
    return warnings


# ---------------- 7. Main pipeline ----------------

def analyze(path, base_year, current_year, min_count=2, use_trigrams=False):
    df = load_data(path)
    result = compare_years(df, base_year, current_year,
                           min_count=min_count, use_trigrams=use_trigrams)
    result = score_phrases(result)
    return result


# ---------------- Test ----------------

if __name__ == "__main__":
    result = analyze("archive/patent.csv", base_year=2010, current_year=2019,
                     min_count=2, use_trigrams=False)
    print(result.head(20))