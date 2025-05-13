def distance_to_score(
    distance: float,
    max_distance: float = 2.0,
    min_score: float = 0.0,
    max_score: float = 100.0
) -> float:
    """
    Convert an L2 distance into a relevance score between min_score and max_score.
    
    A distance of 0 → max_score, a distance ≥ max_distance → min_score,  
    linear interpolation in between.
    
    Args:
        distance: The L2 distance from the query to a document embedding.
        max_distance: The distance at or above which score bottoms out to min_score.
        min_score: Minimum score (default 0.0).
        max_score: Maximum score (default 100.0).
        
    Returns:
        A float relevance score in [min_score, max_score].
    """
    # clamp distance into [0, max_distance]
    d = max(0.0, min(distance, max_distance))
    # invert & scale
    ratio = 1.0 - (d / max_distance)
    score = min_score + ratio * (max_score - min_score)
    return round(score, 2)