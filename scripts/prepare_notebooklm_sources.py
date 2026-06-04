from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "notebooklm_sources"


def export_sources() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_files = sorted(DATA_DIR.glob("*.txt"))
    if not source_files:
        raise SystemExit(f"Nenhum .txt encontrado em {DATA_DIR}")

    index_lines = ["# Fontes para NotebookLM", ""]
    for source in source_files:
        title = source.stem.replace("_", " ").title()
        content = source.read_text(encoding="utf-8").strip()
        target = OUTPUT_DIR / f"{source.stem}.md"
        target.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
        index_lines.append(f"- {target.name}")

    (OUTPUT_DIR / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Arquivos prontos para upload em: {OUTPUT_DIR}")


if __name__ == "__main__":
    export_sources()
