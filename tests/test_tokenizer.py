"""Tests for src/tokenizer.py."""

from src.tokenizer import split_identifier, tokenize


def test_empty_and_punctuation_only() -> None:
    assert tokenize("") == []
    assert tokenize("   \n\t") == []
    assert tokenize("... ---> ()") == []


def test_lowercase_and_separators() -> None:
    assert tokenize("How do I enable lora?") == [
        "how", "do", "i", "enable", "lora",
    ]
    assert tokenize("vllm.engine.arg_utils") == [
        "vllm", "engine", "arg_utils", "arg", "utils",
    ]


def test_snake_case_keeps_whole_and_pieces() -> None:
    assert tokenize("get_model_config") == [
        "get_model_config", "get", "model", "config",
    ]


def test_camel_case_keeps_whole_and_pieces() -> None:
    assert tokenize("LLMEngine") == ["llmengine", "llm", "engine"]
    assert tokenize("AsyncLLMEngine") == [
        "asyncllmengine", "async", "llm", "engine",
    ]


def test_single_piece_word_is_not_duplicated() -> None:
    assert tokenize("Engine") == ["engine"]
    assert tokenize("Qwen3") == ["qwen3"]
    assert tokenize("_private") == ["_private", "private"]


def test_repetitions_are_kept() -> None:
    assert tokenize("lora lora Lora") == ["lora", "lora", "lora"]


def test_mixed_case_word_is_split_like_camel_case() -> None:
    # LoRA looks like CamelCase to the regex: the pieces are noise, but
    # the whole word is kept so an exact match on "lora" still works.
    assert tokenize("LoRA") == ["lora", "lo", "ra"]


def test_split_identifier() -> None:
    assert split_identifier("max_model_len") == ["max", "model", "len"]
    assert split_identifier("HTTPServer") == ["http", "server"]
    assert split_identifier("fp8") == ["fp8"]
