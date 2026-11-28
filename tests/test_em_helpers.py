import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoramento_em import should_accept_expanded_sam_mask  # noqa: E402


@pytest.mark.parametrize(
    ("area_box", "area_pt", "inter_with_box", "frame_area", "expected"),
    [
        (1000, 1500, 800, 10000, True),
        (1000, 900, 800, 10000, False),
        (1000, 1050, 800, 10000, False),
        (1000, 1051, 800, 10000, True),
        (1000, 3800, 800, 20000, True),
        (1000, 3801, 800, 20000, False),
        (1000, 2400, 800, 10000, True),
        (1000, 2401, 800, 10000, False),
        (1000, 1500, 700, 10000, True),
        (1000, 1500, 699, 10000, False),
        (0, 100, 0, 10000, False),
    ],
    ids=[
        "valid-expanded-mask",
        "smaller-expanded-mask",
        "boundary-exact-5pct-expansion-rejected",
        "boundary-just-above-5pct-expansion-accepted",
        "boundary-exact-3-8x-box-area-accepted",
        "boundary-above-3-8x-box-area-rejected",
        "boundary-exact-24pct-frame-area-accepted",
        "boundary-above-24pct-frame-area-rejected",
        "boundary-exact-70pct-overlap-accepted",
        "boundary-below-70pct-overlap-rejected",
        "zero-original-mask-area-safeguard",
    ],
)
def test_should_accept_expanded_sam_mask(area_box, area_pt, inter_with_box, frame_area, expected):
    assert should_accept_expanded_sam_mask(area_box, area_pt, inter_with_box, frame_area) is expected
