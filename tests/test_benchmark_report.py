import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rodar_suite_benchmark as benchmark


def test_build_benchmark_records_keeps_latest_run_and_missing_experiments():
    df = pd.DataFrame([
        {"run_id": "old_tag", "frames_processados": 12},
        {"run_id": "new_tag", "pasta": "runs/new_tag", "frames_processados": 40,
         "peixes_contados": 0, "tempo_execucao_s": 1.5, "comprimento_mediano_px": 0},
        {"run_id": "new_tag_extra", "frames_processados": 99},
    ])
    records = benchmark.build_benchmark_records(df, [("missing", "Sem dados", "ERRO(1)"), ("tag", "Cena", "OK")])
    assert records == [
        {"tag": "missing", "description": "Sem dados", "status": "ERRO(1)", "frames_processados": None,
         "peixes_contados": None, "tempo_execucao_s": None, "comprimento_mediano_px": None,
         "erro_medio_pct": None, "pasta": None},
        {"tag": "tag", "description": "Cena", "status": "OK", "frames_processados": 40,
         "peixes_contados": 0, "tempo_execucao_s": 1.5, "comprimento_mediano_px": 0,
         "erro_medio_pct": None, "pasta": "runs/new_tag"},
    ]


def test_report_only_exports_same_mean_error_as_markdown(tmp_path, monkeypatch):
    tag, description, _ = benchmark.EXPERIMENTS[0]
    run = tmp_path / ("latest_" + tag)
    run.mkdir()
    pd.DataFrame([{
        "run_id": run.name, "pasta": str(run), "frames_processados": 30, "peixes_contados": 2,
        "comprimento_mediano_px": 120.5, "tempo_execucao_s": 2.1,
    }]).to_csv(tmp_path / "indice_experimentos.csv", index=False)
    pd.DataFrame({"erro_estimado_pct": [4.5, 5.5]}).to_csv(run / "resumo_peixes.csv", index=False)
    output = tmp_path / "comparativo.json"
    monkeypatch.setattr(benchmark, "EXP_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--report-only", "--json-out", str(output)])
    benchmark.main()
    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(records) == len(benchmark.EXPERIMENTS)
    assert records[0]["description"] == description
    assert records[0]["erro_medio_pct"] == 5.0
    assert records[1]["erro_medio_pct"] is None
    assert "±5.0%" in (tmp_path / "comparativo_10_experimentos.md").read_text(encoding="utf-8")


def test_report_only_respects_only_filter(tmp_path, monkeypatch):
    first_tag, _, _ = benchmark.EXPERIMENTS[0]
    selected_tag, selected_desc, _ = benchmark.EXPERIMENTS[1]
    run_first = tmp_path / ("latest_" + first_tag)
    run_selected = tmp_path / ("latest_" + selected_tag)
    run_first.mkdir()
    run_selected.mkdir()
    pd.DataFrame([
        {
            "run_id": run_first.name, "pasta": str(run_first), "frames_processados": 30,
            "peixes_contados": 2, "comprimento_mediano_px": 120.5, "tempo_execucao_s": 2.1,
        },
        {
            "run_id": run_selected.name, "pasta": str(run_selected), "frames_processados": 45,
            "peixes_contados": 3, "comprimento_mediano_px": 135.0, "tempo_execucao_s": 3.4,
        },
    ]).to_csv(tmp_path / "indice_experimentos.csv", index=False)
    output = tmp_path / "comparativo.json"
    monkeypatch.setattr(benchmark, "EXP_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--report-only", "--only", selected_tag, "--json-out", str(output)]
    )
    benchmark.main()
    records = json.loads(output.read_text(encoding="utf-8"))
    assert [r["tag"] for r in records] == [selected_tag]
    assert records[0]["description"] == selected_desc
    md_text = (tmp_path / "comparativo_10_experimentos.md").read_text(encoding="utf-8")
    assert f"`{selected_tag}`" in md_text
    assert f"`{first_tag}`" not in md_text
