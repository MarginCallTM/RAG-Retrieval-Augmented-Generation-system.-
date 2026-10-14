"""Tests for src/tokenizer.py."""

from src.tokenizer import split_identifier, tokenize


def test_empty_and_punctuation_only() -> None:
    assert tokenize("") == []
    assert tokenize("   \n\t") == []
    assert tokenize("... ---> ()") == []


def test_lowercase_and_separators() -> None:
    assert tokenize("Enable LORA Adapters?") == ["enable", "lora", "adapters"]
    assert tokenize("vllm.engine.arg_utils") == [
        "vllm", "engine", "arg_utils", "arg", "utils",
    ]


def test_stopwords_are_dropped_everywhere() -> None:
    # Function words mark "this is prose" on a code-heavy corpus and
    # pushed Markdown chunks up on every question (stage 6, +3 pts docs).
    assert tokenize("How do I enable lora?") == ["enable", "lora"]
    assert tokenize("What is the") == []
    # Also inside identifier pieces: "APIs" splits into "ap" + "is".
    assert tokenize("APIs") == ["apis", "ap"]


def test_one_char_pieces_are_dropped() -> None:
    # "v" and "e" are rare, so BM25 gives them a high weight for nothing.
    # The whole word always stays; two-letter pieces such as "kv" stay.
    assert tokenize("vLLM") == ["vllm", "llm"]
    assert tokenize("MoE") == ["moe", "mo"]
    assert tokenize("kv_cache") == ["kv_cache", "kv", "cache"]


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
