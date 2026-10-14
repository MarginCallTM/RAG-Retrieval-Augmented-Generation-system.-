"""Generate a grounded answer from retrieved chunks with a local model.

The model is loaded lazily, only by the ``answer*`` commands: ``index`` and
``search`` never import torch. The prompt building and the context budget
are pure functions so that they are tested without the model.
"""

from typing import Any, Callable, Dict, List, Sequence

from src.models import Chunk

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
# Tokens of sources put in the prompt. 5 chunks of 2000 characters are
# about 2 500 tokens; prefill time on CPU grows with this number.
DEFAULT_CONTEXT_TOKENS = 3000
# Enough for a few sentences and a citation, not for an essay.
DEFAULT_MAX_NEW_TOKENS = 256
NO_ANSWER = "I don't know."

SYSTEM_PROMPT = (
    "You answer questions about the vLLM code base.\n"
    "Rules:\n"
    "- Use only the numbered sources below. Do not use prior knowledge.\n"
    f"- If the sources do not contain the answer, reply exactly: {NO_ANSWER}\n"
    "- When the answer is a command, a code snippet or a value, copy it "
    "in full from the source, inside a code block. Never shorten it "
    "with '...'.\n"
    "- Cite the sources you used as [1], [2].\n"
    "- Be concise: a short explanation plus the quoted code if any."
)


def fit_sources(
    chunks: Sequence[Chunk],
    budget: int,
    count_tokens: Callable[[str], int],
    truncate: Callable[[str, int], str],
) -> List[Chunk]:
    """Keep the chunks, in rank order, that fit in ``budget`` tokens.

    The list stops at the first chunk that does not fit, so the sources
    given to the model are always the best-ranked ones. If even the first
    chunk is too long, it is truncated rather than dropped: a cut source
    beats no source.

    Args:
        chunks: Retrieved chunks, best first.
        budget: Maximum number of tokens for all the sources together.
        count_tokens: Token count of a text, from the model's tokenizer.
        truncate: Cut a text to a number of tokens.

    Returns:
        The chunks to put in the prompt, possibly empty.
    """
    kept: List[Chunk] = []
    used = 0
    for chunk in chunks:
        tokens = count_tokens(chunk.text)
        if used + tokens <= budget:
            kept.append(chunk)
            used += tokens
            continue
        if not kept and budget > 0:
            kept.append(chunk.model_copy(
                update={"text": truncate(chunk.text, budget)}
            ))
        break
    return kept


def build_messages(
    question: str, chunks: Sequence[Chunk]
) -> List[Dict[str, str]]:
    """Return the chat messages: the rules, then the sources and question.

    Each source is numbered and labelled with its file path, so the model
    can cite it and the reader can check the citation.
    """
    sources = "\n\n".join(
        f"[{number}] {chunk.file_path}\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
    if not sources:
        sources = "(no source)"
    user = f"Sources:\n\n{sources}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    """A local causal language model answering from given chunks.

    Loading takes seconds and 2.4 GB of RAM (0.6 B parameters in float32),
    so one instance serves every question of a dataset.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        """Load the tokenizer and the model, on the GPU if there is one.

        Args:
            model_name: A Hugging Face model id or a local directory.
            context_tokens: Token budget for the sources in the prompt.
            max_new_tokens: Maximum length of the generated answer.

        Raises:
            OSError: If the model cannot be downloaded or read.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.context_tokens = context_tokens
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        """Number of model tokens in ``text``."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, tokens: int) -> str:
        """Cut ``text`` to its first ``tokens`` model tokens."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:tokens]
        return str(self.tokenizer.decode(ids))

    def select(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        """The chunks that fit the context budget, see ``fit_sources``."""
        return fit_sources(
            chunks, self.context_tokens, self.count_tokens, self.truncate
        )

    def generate(self, question: str, chunks: Sequence[Chunk]) -> str:
        """Answer ``question`` from ``chunks`` only.

        Decoding is greedy, so the same inputs give the same answer. The
        Qwen3 "thinking" mode is disabled: on CPU it would spend hundreds
        of tokens before the first word of the answer.

        Args:
            question: The user's question.
            chunks: The sources, already selected with ``select``.

        Returns:
            The answer text, stripped. Never empty: ``NO_ANSWER`` at worst.
        """
        import torch

        prompt = self.tokenizer.apply_chat_template(
            build_messages(question, chunks),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated: Any = output[0][inputs["input_ids"].shape[1]:]
        text = str(self.tokenizer.decode(generated, skip_special_tokens=True))
        return text.strip() or NO_ANSWER
