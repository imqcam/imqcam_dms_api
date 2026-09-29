from pathlib import Path
import ast
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def sanitize_cell(source):
    lines = []

    for line in source.splitlines():
        stripped = line.lstrip()

        if stripped.startswith(("!", "%", "?")):
            lines.append("# skipped notebook magic")
        else:
            lines.append(line)

    return "\n".join(lines)


def validate_notebook(path):
    errors = []

    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)

    code = "\n".join(
        sanitize_cell(cell.source)
        for cell in nb.cells
        if cell.cell_type == "code"
    )

    # Catch the filtered_df bug from the review.
    if re.search(r"\bfiltered_df\b", code):
        assigned = bool(
            re.search(
                r"(?m)^\s*filtered_df\s*=",
                code
            )
        )

        if not assigned:
            errors.append(
                f"{path}: filtered_df is referenced "
                "but never defined"
            )

    # Basic Python syntax check.
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        try:
            ast.parse(
                sanitize_cell(cell.source)
            )
        except SyntaxError as exc:
            errors.append(
                f"{path}: syntax error in cell "
                f"{index}: {exc}"
            )

    return errors


def main():
    notebooks = sorted(
        path
        for path in ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )

    errors = []

    for notebook in notebooks:
        errors.extend(
            validate_notebook(notebook)
        )

    if errors:
        for error in errors:
            print(error)

        raise SystemExit(1)

    print(
        f"Validated {len(notebooks)} notebook(s)."
    )


if __name__ == "__main__":
    main()