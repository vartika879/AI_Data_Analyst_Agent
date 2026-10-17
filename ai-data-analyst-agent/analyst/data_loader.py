"""Safe CSV loading and validation.

The loader takes raw bytes (not a Streamlit object), so it can be tested
without Streamlit. Every failure becomes a DataLoadError with a message
that is safe to show directly to the user.
"""

import io

import pandas as pd


class DataLoadError(ValueError):
    """Raised when an uploaded file cannot be turned into a usable DataFrame."""


def _decode(file_bytes: bytes) -> str:
    # utf-8-sig handles normal UTF-8 and also removes the BOM that Excel adds.
    try:
        return file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        # latin-1 can decode any byte, so this is a last-resort fallback
        # for older Excel/Windows exports.
        return file_bytes.decode("latin-1")


def load_csv(file_bytes: bytes, filename: str, max_bytes: int) -> pd.DataFrame:
    """Validate raw upload bytes and return a DataFrame."""
    if not filename.lower().endswith(".csv"):
        raise DataLoadError("Only .csv files are supported.")

    if len(file_bytes) == 0:
        raise DataLoadError("The uploaded file is empty.")

    if len(file_bytes) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise DataLoadError(f"File is too large. The limit is {limit_mb:g} MB.")

    # Real CSVs are text. Null bytes mean a binary file renamed to .csv.
    if b"\x00" in file_bytes[:8192]:
        raise DataLoadError("This does not look like a text CSV file.")

    text = _decode(file_bytes)
    if text.strip() == "":
        raise DataLoadError("The uploaded file contains no data.")

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        raise DataLoadError("The uploaded file contains no data.") from None
    except pd.errors.ParserError:
        raise DataLoadError(
            "Could not parse the file as CSV. Check that every row has the "
            "same number of columns."
        ) from None

    if df.shape[1] == 0:
        raise DataLoadError("The CSV has no columns.")
    if len(df) == 0:
        raise DataLoadError("The CSV has column headers but no data rows.")

    # Clean column names so later column validation is predictable.
    df.columns = [str(col).strip() for col in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise DataLoadError("The CSV has duplicate column names.")

    return df