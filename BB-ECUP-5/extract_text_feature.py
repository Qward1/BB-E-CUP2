
import re
import string
import pandas as pd

# =============================
#  Общие словари
# =============================
SUSPICIOUS_KEYWORDS = [
    # русский
    "копия", "копии", "копией", "копий", "коп.", "коп",
    "реплика", "реплики", "репликой", "репл.", "репл",
    "подделка", "подделки", "подделкой",
    "аналог", "аналоги",
    # английский
    "replica", "replika", "rep.",
    "copy", "copi",
    "fake", "fakes", "facke",
    "1:1", "1/1", "aaa", "aa", "7a", "top quality", "100% original"
]

# =============================
#  Вспомогательные функции
# =============================
def count_punctuation(text: str) -> int:
    """Количество знаков пунктуации"""
    if not isinstance(text, str) or not text.strip():
        return 0
    return sum(1 for ch in text if ch in string.punctuation)

def en_ru_ratio(text: str) -> float:
    """Соотношение английских и русских символов"""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    en = len(re.findall(r"[a-zA-Z]", text))
    ru = len(re.findall(r"[а-яА-Я]", text))
    return en / ru if ru > 0 else (float(en) if en > 0 else 0.0)

# =============================
# Универсальная функция
# =============================
def prepare_text_features(df: pd.DataFrame, cols=["description", "name_rus"]) -> pd.DataFrame:


    # --- очистка текстов ---
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return text
        text = re.sub(r"<.*?>", " ", text)       # убираем html-теги
        text = re.sub(r"\s+", " ", text)         # убираем множественные пробелы
        return text.strip()

    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # --- description ---
    desc_features = []
    for text in df["description"]:
        if not isinstance(text, str) or not text.strip():
            desc_features.append({
                "desc_char_count": 0,
                "desc_word_count": 0,
                "desc_spec_symbol_count": 0,
                "desc_number_count": 0,
                "desc_punct_count": 0,
                "desc_has_keywords": 0,
                "desc_en_ru_ratio": 0.0
            })
            continue

        desc_features.append({
            "desc_char_count": len(text),
            "desc_word_count": len(text.split()),
            "desc_spec_symbol_count": len(re.findall(r"[!@#$%^&*()_+=\[\]{};:\"'|\\<>,.?/~`]", text)),
            "desc_number_count": len(re.findall(r"\d+", text)),
            "desc_punct_count": count_punctuation(text),
            "desc_has_keywords": int(any(kw in text.lower() for kw in SUSPICIOUS_KEYWORDS)),
            "desc_en_ru_ratio": en_ru_ratio(text)
        })

    desc_df = pd.DataFrame(desc_features)

    # --- name_rus ---
    name_features = []
    for text in df["name_rus"]:
        if not isinstance(text, str) or not text.strip():
            name_features.append({
                "name_length": 0,
                "name_has_keywords": 0,
                "name_upper_count": 0,
                "name_full_caps_words": 0,
                "name_en_ru_ratio": 0.0,
                "name_punct_count": 0
            })
            continue

        words = text.split()
        name_features.append({
            "name_length": len(words),
            "name_has_keywords": int(any(kw in text.lower() for kw in SUSPICIOUS_KEYWORDS)),
            "name_upper_count": sum(1 for c in text if c.isupper()),
            "name_full_caps_words": sum(1 for w in words if w.isupper()),
            "name_en_ru_ratio": en_ru_ratio(text),
            "name_punct_count": count_punctuation(text)
        })

    name_df = pd.DataFrame(name_features)

    #  объединение
    features = pd.concat([desc_df, name_df], axis=1)

    # удаляем ненужные
    features = features[['desc_has_keywords', 'desc_spec_symbol_count']]

    return features
