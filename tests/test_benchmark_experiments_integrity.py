import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rodar_suite_benchmark as benchmark


def test_experiments_has_exactly_ten_entries():
    assert len(benchmark.EXPERIMENTS) == 10


def test_experiment_tags_are_unique():
    tags = [tag for tag, _, _ in benchmark.EXPERIMENTS]
    assert len(tags) == len(set(tags)), "experiment tags must be unique"


def test_experiment_descriptions_are_non_empty():
    for tag, description, _ in benchmark.EXPERIMENTS:
        assert isinstance(description, str) and description.strip(), (
            f"experiment {tag!r} must have a non-empty description"
        )


def test_experiment_frame_ranges_are_valid():
    for tag, _, args in benchmark.EXPERIMENTS:
        assert "--start-frame" in args, f"experiment {tag!r} must declare --start-frame"
        assert "--end-frame" in args, f"experiment {tag!r} must declare --end-frame"
        start = int(args[args.index("--start-frame") + 1])
        end = int(args[args.index("--end-frame") + 1])
        assert start < end, f"experiment {tag!r} must have --start-frame < --end-frame"
