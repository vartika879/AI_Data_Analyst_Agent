import pytest

from analyst.data_loader import DataLoadError, load_csv

LIMIT = 1024 * 1024  # 1 MB


def test_valid_csv_loads():
    df = load_csv(b"name,age\nAsha,30\nRavi,25\n", "people.csv", LIMIT)
    assert list(df.columns) == ["name", "age"]
    assert len(df) == 2
    assert df["age"].tolist() == [30, 25]


def test_column_names_are_stripped():
    df = load_csv(b" a , b \n1,2\n", "x.csv", LIMIT)
    assert list(df.columns) == ["a", "b"]


def test_empty_file_raises():
    with pytest.raises(DataLoadError, match="empty"):
        load_csv(b"", "x.csv", LIMIT)


def test_whitespace_only_file_raises():
    with pytest.raises(DataLoadError, match="no data"):
        load_csv(b"   \n\n  ", "x.csv", LIMIT)


def test_header_only_file_raises():
    with pytest.raises(DataLoadError, match="no data rows"):
        load_csv(b"a,b\n", "x.csv", LIMIT)


def test_file_over_size_limit_raises():
    with pytest.raises(DataLoadError, match="too large"):
        load_csv(b"a,b\n1,2\n", "x.csv", max_bytes=5)


def test_non_csv_extension_raises():
    with pytest.raises(DataLoadError, match=r"\.csv"):
        load_csv(b"a,b\n1,2\n", "data.txt", LIMIT)


def test_extension_check_is_case_insensitive():
    df = load_csv(b"a,b\n1,2\n", "DATA.CSV", LIMIT)
    assert len(df) == 1


def test_binary_content_raises():
    with pytest.raises(DataLoadError, match="text CSV"):
        load_csv(b"\x00\x01\x02binary", "x.csv", LIMIT)


def test_ragged_rows_raise():
    with pytest.raises(DataLoadError, match="parse"):
        load_csv(b"a,b\n1,2\n3,4,5,6\n", "x.csv", LIMIT)


def test_duplicate_column_names_raise():
    with pytest.raises(DataLoadError, match="duplicate"):
        load_csv(b"a, a\n1,2\n", "x.csv", LIMIT)


def test_latin1_file_loads():
    raw = "name\nJosé\n".encode("latin-1")
    df = load_csv(raw, "x.csv", LIMIT)
    assert df["name"].tolist() == ["José"]


def test_utf8_bom_is_removed_from_first_column():
    raw = "name,age\nAsha,30\n".encode("utf-8-sig")
    df = load_csv(raw, "x.csv", LIMIT)
    assert list(df.columns) == ["name", "age"]