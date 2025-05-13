import pytest
from src.utils.scoring import distance_to_score

def test_distance_zero_returns_max_score():
    # distance 0.0 should map to 100.0 by default
    assert distance_to_score(0.0) == 100.0

def test_distance_equal_max_distance_returns_min_score():
    # distance == max_distance (2.0) should map to 0.0 by default
    assert distance_to_score(2.0) == 0.0

def test_distance_greater_than_max_distance_clamps_to_min_score():
    # anything above max_distance clamps to min_score
    assert distance_to_score(5.0) == 0.0

def test_midpoint_distance_scales_linearly():
    # at half of max_distance: 1.0 → 50.0
    assert distance_to_score(1.0) == 50.0

def test_custom_max_distance_and_scores():
    # with max_distance=10, range 0–10 maps to 0–1
    score = distance_to_score(2.5, max_distance=10.0, min_score=0.0, max_score=1.0)
    # 1 - (2.5/10) = 0.75 → rounded to 0.75
    assert score == pytest.approx(0.75)

def test_custom_score_bounds():
    # custom score range 10–20
    score_low = distance_to_score(2.0, max_distance=2.0, min_score=10.0, max_score=20.0)
    assert score_low == 10.0  # at max_distance
    
    score_high = distance_to_score(0.0, max_distance=2.0, min_score=10.0, max_score=20.0)
    assert score_high == 20.0  # at zero distance

def test_negative_distance_clamps_to_max_score():
    # negative distances treated as zero
    assert distance_to_score(-1.0) == 100.0
