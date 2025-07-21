#!/usr/bin/env python3
"""
extract_code.py

Usage:
    python extract_code.py path/to/notebook.ipynb path/to/output.py
"""
import sys
import nbformat


def extract_code_cells(ipynb_path: str, py_path: str) -> None:
    # Read the notebook
    nb = nbformat.read(ipynb_path, as_version=4)

    # Open the output file
    with open(py_path, "w", encoding="utf-8") as fout:
        for cell in nb.cells:
            if cell.cell_type == "code":
                # Write the source of each code cell
                fout.write(cell.source.rstrip() + "\n\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_code.py notebook.ipynb output.py")
        sys.exit(1)

    ipynb_file = sys.argv[1]
    py_file = sys.argv[2]
    extract_code_cells(ipynb_file, py_file)
    print(f"Extracted code cells from {ipynb_file} → {py_file}")
