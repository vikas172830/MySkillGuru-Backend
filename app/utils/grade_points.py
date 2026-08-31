GRADE_POINTS = {
    "A+": 10,
    "S/A+": 10,  # alias for compatibility with CGPA sample Excel
    "A": 9.5,
    "A-": 9,
    "B+": 8.5,
    "B": 8,
    "B-": 7.5,
    "C+": 7,
    "C": 6.5,
    "C-": 6,
    "D": 5,
    "U": 0,
}


def get_grade_point(grade: str) -> float:
    """Falls back to 0 for an unrecognized/unmapped grade so calculations don't break silently."""
    if not grade:
        return 0
    return GRADE_POINTS.get(grade.strip(), 0)
