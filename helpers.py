def clean_str(s: str) -> str:
    """
    Replace any lone surrogate code points with the U+FFFD
    replacement char so Chroma/Rust can encode to UTF‑8.
    """
    return s.encode("utf-8", "surrogatepass").decode("utf-8", "replace")