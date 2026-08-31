from typing import Any, Dict, List


def assign_relative_grades(
    students: List[Dict[str, Any]],
    grading_config: List[Dict[str, Any]],
    max_marks: float = 100,
    pass_percentage: float = 30,
) -> List[Dict[str, Any]]:
    """
    Faithful port of the reference workbook's live Excel formula
    ("CGPA Calculation Sample 1.xlsx"):

        =IF(marks < pass_floor, "U",
            IFS(percentile<band1, grade1, percentile<band2, grade2, ...))
        where percentile = (RANK.EQ(marks, ALL students, descending) - 1)
                            / COUNTA(ALL students)

    Two things this deliberately does NOT do, because the Excel formula
    doesn't either:
      - It does not remove failing students before computing percentiles.
        RANK.EQ/COUNTA in the sheet cover every student in the row,
        including ones who will end up graded U — a failing student still
        counts toward everyone else's rank and percentile.
      - It does not avoid splitting a tied group across two grades. Each
        student's grade is evaluated independently from their own
        percentile; two students with identical marks always land on the
        exact same grade as a natural consequence (same rank -> same
        percentile -> same band), never because of extra tie-handling
        logic.

    students must already be sorted highest to lowest by overall_total.
    grading_config should be ordered highest grade first, for example:
    [{"grade": "A+", "percentage": 5}, {"grade": "A", "percentage": 8}].
    """
    if not students:
        return students

    total_students = len(students)  # Excel's COUNTA — every student, pass or fail
    all_totals = [student.get("overall_total", 0) or 0 for student in students]

    cumulative_percentage = 0.0
    bands = []  # [(cutoff_fraction, grade), ...], highest grade first
    for config in grading_config:
        grade = config.get("grade")
        if grade == "U":
            continue
        try:
            percentage = float(config.get("percentage", 0) or 0)
        except (TypeError, ValueError):
            percentage = 0
        if not grade or percentage <= 0:
            continue
        cumulative_percentage += percentage
        bands.append((cumulative_percentage / 100, grade))

    if not bands:
        raise ValueError("At least one relative grading percentage must be greater than 0")

    # Guard against the configured percentages summing to slightly under
    # 100 (floating point) by making the lowest band an explicit catch-all,
    # matching the formula's final "TRUE, <lowest grade>" branch.
    bands[-1] = (max(bands[-1][0], 1.0), bands[-1][1])

    for student in students:
        overall_total = student.get("overall_total", 0) or 0
        percentage_of_max = (overall_total / max_marks * 100) if max_marks else 0

        if percentage_of_max < pass_percentage:
            student["grade"] = "U"
            continue

        rank = 1 + sum(1 for value in all_totals if value > overall_total)
        percentile = (rank - 1) / total_students

        for cutoff, grade in bands:
            if percentile < cutoff:
                student["grade"] = grade
                break

    return students
