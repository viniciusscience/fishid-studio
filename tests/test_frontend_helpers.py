"""Unit tests for frontend TRANSLATIONS key parity across pt, en, and es."""

import ast
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "frontend" / "app.py"


def _load_translations() -> dict[str, dict[str, str]]:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "TRANSLATIONS":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "TRANSLATIONS":
                    return ast.literal_eval(node.value)
    raise AssertionError("TRANSLATIONS dictionary not found in frontend/app.py")


def test_translations_contain_required_languages() -> None:
    translations = _load_translations()
    assert {"pt", "en", "es"}.issubset(translations.keys())


def test_translations_key_parity_across_pt_en_es() -> None:
    translations = _load_translations()
    pt_keys = set(translations["pt"].keys())
    en_keys = set(translations["en"].keys())
    es_keys = set(translations["es"].keys())

    assert pt_keys == en_keys, f"pt vs en mismatch: {pt_keys ^ en_keys}"
    assert pt_keys == es_keys, f"pt vs es mismatch: {pt_keys ^ es_keys}"
    assert len(pt_keys) > 0


def test_translations_values_are_non_empty_strings() -> None:
    translations = _load_translations()
    for lang in ("pt", "en", "es"):
        for key, value in translations[lang].items():
            assert isinstance(value, str), f"{lang}[{key!r}] is not a str"
            assert value.strip(), f"{lang}[{key!r}] is empty"
