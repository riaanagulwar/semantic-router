import pytest

from app.router import InputRejected, MAX_QUERY_LEN, validate_input


def test_empty_string_rejected():
    with pytest.raises(InputRejected):
        validate_input("")


def test_whitespace_only_rejected():
    with pytest.raises(InputRejected):
        validate_input("   \n\t  ")


def test_oversized_query_rejected():
    with pytest.raises(InputRejected):
        validate_input("a" * (MAX_QUERY_LEN + 1))


def test_max_length_query_accepted():
    text = "a" * MAX_QUERY_LEN
    assert validate_input(text) == text


def test_control_chars_stripped():
    assert validate_input("hello\x00\x07world") == "helloworld"


def test_normal_unicode_passes_through():
    text = "¿Cuál es mi saldo? 你好 😀"
    assert validate_input(text) == text


def test_surrounding_whitespace_trimmed():
    assert validate_input("  hello  ") == "hello"
