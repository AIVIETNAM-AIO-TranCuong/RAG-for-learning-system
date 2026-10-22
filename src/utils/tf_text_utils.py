import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import unicodedata

try:
    import tensorflow as tf
    # Đảm bảo TensorFlow không chiếm dụng VRAM của PyTorch trên GPU
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    HAS_TF = True
except ImportError:
    HAS_TF = False


def normalize_unicode(text: str) -> str:
    if HAS_TF:
        tensor = tf.constant(text, dtype=tf.string)
        normalized = tf.strings.unicode_transcode(
            tensor,
            input_encoding="UTF-8",
            output_encoding="UTF-8",
            errors="ignore",    
        )
        return normalized.numpy().decode("utf-8")
    else:
        return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    if HAS_TF:
        tensor = tf.constant(text, dtype=tf.string)
        cleaned = tf.strings.regex_replace(tensor, r"[ \t]+", " ")
        cleaned = tf.strings.regex_replace(cleaned, r"\n{3,}", "\n\n")
        cleaned = tf.strings.regex_replace(cleaned, r"[ \t]*\n[ \t]*", "\n")
        return cleaned.numpy().decode("utf-8").strip()
    else:
        t = re.sub(r"[ \t]+", " ", text)
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
        return t.strip()


def remove_pdf_artifacts(text: str) -> str:
    if HAS_TF:
        tensor = tf.constant(text, dtype=tf.string)
        cleaned = tf.strings.regex_replace(tensor, r"[-–]\s*\d+\s*[-–]", "")
        cleaned = tf.strings.regex_replace(cleaned, r"[\x00-\x08\x0b-\x1f\x7f]", "")
        cleaned = tf.strings.regex_replace(cleaned, r"[─━═]{3,}", "")
        return cleaned.numpy().decode("utf-8")
    else:
        t = re.sub(r"[-–]\s*\d+\s*[-–]", "", text)
        t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", t)
        t = re.sub(r"[─━═]{3,}", "", t)
        return t


def preprocess_pdf_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_pdf_artifacts(text)
    text = clean_whitespace(text)
    return text


def preprocess_batch(texts: list[str]) -> list[str]:
    return [preprocess_pdf_text(t) for t in texts]
