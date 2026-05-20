import io

from pypdf import PdfReader

ALLOWED_EXTS = {
    "pdf",
    "txt", "md", "markdown",
    "csv", "tsv",
    "json", "yaml", "yml", "toml",
    "log",
    "py", "js", "ts", "jsx", "tsx", "html", "css", "go", "rs", "java", "c", "cpp", "h", "sh",
}
MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
MAX_CHARS = 200_000  # cap extracted text per file


class AttachmentError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    if len(data) > MAX_BYTES:
        raise AttachmentError(
            f"{filename}: too large ({len(data):,} bytes; max {MAX_BYTES:,})"
        )
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        raise AttachmentError(
            f"{filename}: unsupported file type (.{ext or 'unknown'}). "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
        )
    if ext == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            raise AttachmentError(f"{filename}: PDF parse failed ({e})") from e
    else:
        text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_CHARS:
        text = (
            text[:MAX_CHARS]
            + f"\n\n[truncated by server — original was {len(text):,} characters]"
        )
    return text
