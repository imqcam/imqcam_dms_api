from pathlib import Path
import ast
import io

import nbformat
from pyflakes.api import check
from pyflakes.reporter import Reporter


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "04_file_collections.ipynb"


def sanitize_cell(source):
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%", "?")):
            lines.append("# skipped notebook magic/shell command")
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_notebook(path):
    errors = []

    if not path.exists():
        return [f"{path}: notebook not found"]

    try:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
    except Exception as exc:
        return [f"{path}: invalid notebook structure: {exc}"]

    code_cells = []

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = sanitize_cell(cell.source)

        try:
            ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"{path}: syntax error in code cell {index}: {exc}")

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    f"{path}: saved error output in code cell {index}: "
                    f"{output.get('ename')}: {output.get('evalue')}"
                )

        code_cells.append(source)

    combined_code = "\n\n".join(code_cells)

    stdout = io.StringIO()
    stderr = io.StringIO()
    reporter = Reporter(stdout, stderr)
    check(combined_code, str(path), reporter=reporter)

    messages = [
        line.strip()
        for line in (stdout.getvalue() + stderr.getvalue()).splitlines()
        if line.strip()
    ]

    messages = [
        msg for msg in messages
        if "undefined name 'display'" not in msg
    ]

    errors.extend(messages)
    return errors


def main():
    errors = validate_notebook(NOTEBOOK)

    if errors:
        print("Notebook validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Validated {NOTEBOOK.relative_to(ROOT)} successfully.")


if __name__ == "__main__":
    main()
