from __future__ import annotations

import io
import zipfile
from pathlib import Path

from agent_challenge.analyzer.ast_features import (
    AST_STATUS_OK,
    AST_STATUS_SYNTAX_ERROR,
    build_python_ast_feature_rows,
    extract_python_ast_features,
)
from agent_challenge.submissions.artifacts import store_zip_bytes

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


def zip_bytes(entries: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    archive_entries = {"agent.py": ENTRYPOINT_SOURCE, **entries}
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in archive_entries.items():
            if filename == "agent.py":
                contents = agent_source(contents)
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def extract(entries: dict[str, str | bytes], tmp_path: Path):
    metadata = store_zip_bytes(zip_bytes=zip_bytes(entries), artifact_root=str(tmp_path))
    assert metadata.manifest is not None
    return extract_python_ast_features(
        zip_path=metadata.artifact_path,
        manifest=metadata.manifest,
    )


def test_valid_python_file_returns_counts_hashes_and_rows(tmp_path: Path) -> None:
    report = extract({"agent.py": "def solve():\n    return 1\n"}, tmp_path)

    assert report.status == AST_STATUS_OK
    assert report.reason is None
    by_path = {result.file_path: result for result in report.files}
    assert set(by_path) == {"agent.py"}
    file_result = by_path["agent.py"]
    assert file_result.file_path == "agent.py"
    assert file_result.status == AST_STATUS_OK
    assert file_result.artifact_hash == report.artifact_hash
    assert file_result.file_hash
    assert file_result.ast_hash
    assert file_result.function_count == 1
    assert file_result.class_count == 1
    assert file_result.import_count == 0
    assert file_result.docstring_present is False
    assert file_result.docstring_count == 0
    assert file_result.parser_error is None
    assert "function" in file_result.name_shingles

    rows = build_python_ast_feature_rows(analysis_run_id=42, report=report)

    feature_keys = {row.feature_key for row in rows}
    assert "agent.py:parser_status" in feature_keys
    assert "agent.py:ast_hash" in feature_keys
    assert "agent.py:function_count" in feature_keys
    assert all(row.analysis_run_id == 42 for row in rows)


def test_syntax_invalid_python_records_error_and_continues(tmp_path: Path) -> None:
    report = extract(
        {
            "bad.py": "def bad(:\n    pass\n",
            "good.py": "def solve():\n    return 1\n",
        },
        tmp_path,
    )

    by_path = {result.file_path: result for result in report.files}

    assert report.status == "partial"
    assert by_path["bad.py"].status == AST_STATUS_SYNTAX_ERROR
    assert by_path["bad.py"].parser_error
    assert by_path["bad.py"].syntax_line == 1
    assert by_path["bad.py"].ast_hash is None
    assert by_path["good.py"].status == AST_STATUS_OK
    assert by_path["good.py"].function_count == 1


def test_entrypoint_only_zip_extracts_agent(tmp_path: Path) -> None:
    report = extract({"README.md": "hello\n", "data.json": "{}\n"}, tmp_path)

    assert report.status == AST_STATUS_OK
    assert report.reason is None
    assert [result.file_path for result in report.files] == ["agent.py"]
    assert report.artifact_hash


def test_formatting_comments_import_order_and_variable_names_are_deterministic(
    tmp_path: Path,
) -> None:
    first = extract(
        {
            "agent.py": (
                "import sys\n"
                "import os\n\n"
                "# ignored\n"
                "def solve(value):\n"
                "    helper = value + 1\n"
                "    return helper\n"
            ),
        },
        tmp_path / "first",
    )
    second = extract(
        {
            "agent.py": (
                "import os\n"
                "import sys\n\n"
                "def solve(x):\n\n"
                "    renamed = x + 1  # ignored\n"
                "    return renamed\n"
            ),
        },
        tmp_path / "second",
    )

    first_file = {result.file_path: result for result in first.files}["agent.py"]
    second_file = {result.file_path: result for result in second.files}["agent.py"]
    assert first_file.ast_hash == second_file.ast_hash
    assert first_file.function_count == second_file.function_count == 1
    assert first_file.import_count == second_file.import_count == 2
    assert first_file.name_shingles == second_file.name_shingles
    assert first_file.imports == second_file.imports == ("os", "sys")


def test_docstrings_are_flagged_and_count_as_string_literals(tmp_path: Path) -> None:
    report = extract(
        {
            "agent.py": (
                '"""module docs"""\n\n'
                "class Agent:\n"
                '    """class docs"""\n\n'
                "    def solve(self):\n"
                '        """method docs"""\n'
                "        return 'ok'\n"
            ),
        },
        tmp_path,
    )

    file_result = {result.file_path: result for result in report.files}["agent.py"]
    assert file_result.docstring_present is True
    assert file_result.docstring_count == 3
    assert file_result.string_literal_count == 4
    assert file_result.class_count == 1
    assert file_result.function_count == 1


def test_cache_and_generated_python_paths_are_excluded(tmp_path: Path) -> None:
    report = extract(
        {
            "__pycache__/ignored.py": "def cached():\n    return 1\n",
            ".pytest_cache/ignored.py": "def cached():\n    return 1\n",
            "package/generated_pb2.py": "def generated():\n    return 1\n",
            "agent.py": "def solve():\n    return 1\n",
        },
        tmp_path,
    )

    assert [result.file_path for result in report.files] == ["agent.py"]
