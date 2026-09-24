"""Command definitions exposed through Python fire.

Every public method of: class: 'RagCli' becomes a sub-command of
''uv run python -m src''. The bodies are stubs for now.
"""


class RagCli:
    """Commands of the RAG system, one method per CLI command."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest ''data/raw/'' and build the index under ''data/processed/.

        Args:
                max_chunk_size: Maximum size of a chunk, in characters.
        """

        print(f"index: not implemented (max_chunk_size={max_chunk_size})")

    def search(self, query: str, k: int = 5) -> None:
        """Return the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
        """
        print(f"search: not implemented (query={query!r}, k={k})")

    def search_dataset(
        self, dataset_path: str, k: int = 10, save_directory: str = "."
    ) -> None:
        """Run search over a whole dataset and write a results JSON file.

        Args:
            dataset_path: Path to a dataset JSON file.
            k: number of sources to return per question.
            save_directory: Directory where the results file is written.
            """
        print(
            "search_dataset: not implemented "
            f"(dataset_path={dataset_path!r}, k={k},"
            f"save_directory={save_directory!r})"
        )

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve before answering.
        """
        print(f"answer: not implemented (query= {query!r}, k={k})")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str = "."
    ) -> None:
        """Generate answers for every question of a search results file.

        Args:
            student_search_results_path: Path to a search results JSON file.
            save_directory: Directory where the answers file is written.
        """
        print(
            "answer_dataset: not implemented "
            f"(student_search_result_path={student_search_results_path!r}, "
            f"save_directory={save_directory!r})"
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Report recall@k of a search results file against a dataset.

        Args:
                student_search_results_path: Path to a search result JSON file.
                dataset_path: Path to the ground-truth dataset JSON file.
        """
        print(
            "evaluate: not implemented"
            f"(student_search_results_path={student_search_results_path!r}, "
            f"dataset_path={dataset_path!r})"
        )
