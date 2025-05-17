import os
import re
import sys
from typing import Iterator, List, Tuple

# Patterns considered suspicious or indicative of redundant duplicates
SUSPICIOUS_PATTERNS = [
    r"\\bcopy\\b",
    r"\\bmock\\b",
    r"\\btest\\b",
    r"\\bbackup\\b",
    r" 2",  # filenames containing a space and the number 2
    r"_old\\b",
    r"_bak\\b",
]

# File extensions that are code or configuration and worth scanning
CODE_EXTENSIONS = (
    ".js", ".jsx", ".ts", ".tsx", ".py", ".json", ".sh", ".md", ".yml", ".yaml",
)


def is_suspicious(filename: str) -> bool:
    """Return True if filename matches any suspicious pattern."""
    return any(re.search(pat, filename, re.IGNORECASE) for pat in SUSPICIOUS_PATTERNS)


def iter_files(root: str) -> Iterator[str]:
    """Yield all regular files recursively under *root*."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith(".") or fname.endswith(".pyc"):
                continue
            yield os.path.join(dirpath, fname)


def collect_code_files(root: str) -> List[str]:
    """Return list of code/config files to inspect for references."""
    return [f for f in iter_files(root) if f.endswith(CODE_EXTENSIONS)]


def count_references(target: str, code_files: List[str]) -> int:
    """Return how many times *target* is referenced in *code_files* (excluding itself)."""
    basename = os.path.basename(target)
    stem, _ext = os.path.splitext(basename)
    ref_count = 0
    for code in code_files:
        if code == target:
            continue
        try:
            with open(code, "r", encoding="utf-8", errors="ignore") as fh:
                contents = fh.read()
                if (
                    re.search(rf"\\b{re.escape(stem)}\\b", contents)
                    or basename in contents
                ):
                    ref_count += 1
        except Exception:
            # Skip unreadable files
            continue
    return ref_count


def scan(root: str = ".") -> List[Tuple[str, int, bool]]:
    """Scan *root* and return list of (filepath, reference_count, suspicious_flag)."""
    code_files = collect_code_files(root)
    results: List[Tuple[str, int, bool]] = []
    for fpath in code_files:
        refs = count_references(fpath, code_files)
        results.append((fpath, refs, is_suspicious(os.path.basename(fpath))))
    return results


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    results = scan(root)
    print("Potentially unused or redundant files:\n")
    for path, refs, suspicious in sorted(results, key=lambda x: (x[1], x[0])):
        # Highlight only if zero references or suspicious flag
        if refs == 0 or suspicious:
            print(f"{path} | References: {refs} | Suspicious: {suspicious}")
    print("\nReview the list before deleting — some files may be used indirectly (e.g., entrypoints, configs, CI scripts).")


if __name__ == "__main__":
    main() 