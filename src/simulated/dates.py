"""Accident and claim dates derived deterministically (no randomness) from the claim's real
calendar fields. The data records month, week of month and weekday, not exact dates, so these
are reconstructions: week N is the Monday-start calendar week of the month that contains
day 1 for N = 1; a date that falls outside the month is clamped to the month's first / last
occurrence of that weekday."""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Any, Mapping, Optional

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def week_date(year: int, month: int, week: int, weekday: str) -> Optional[date]:
    if weekday not in WEEKDAYS or not 1 <= month <= 12:
        return None
    wd = WEEKDAYS.index(weekday)
    first = date(year, month, 1)
    week1_monday = first - timedelta(days=first.weekday())
    d = week1_monday + timedelta(days=7 * (int(week) - 1) + wd)
    last = date(year, month, calendar.monthrange(year, month)[1])
    while d < first:
        d += timedelta(days=7)
    while d > last:
        d -= timedelta(days=7)
    return d


def accident_date(c: Mapping[str, Any]) -> Optional[date]:
    try:
        return week_date(int(c["Year"]), MONTHS.index(c["Month"]) + 1, int(c["WeekOfMonth"]), c["DayOfWeek"])
    except (KeyError, ValueError, TypeError):
        return None


def claim_date(c: Mapping[str, Any]) -> Optional[date]:
    """Claim year = accident year, or the next year when the claim month is earlier than the
    accident month (the same assumption as rules V06/V07)."""
    try:
        am, cm = MONTHS.index(c["Month"]) + 1, MONTHS.index(c["MonthClaimed"]) + 1
        year = int(c["Year"]) + (1 if cm < am else 0)
        return week_date(year, cm, int(c["WeekOfMonthClaimed"]), c["DayOfWeekClaimed"])
    except (KeyError, ValueError, TypeError):
        return None


def reporting_lag_days(c: Mapping[str, Any]) -> Optional[int]:
    """Days from accident to claim, measured conservatively: the data fixes the accident and
    claim only to a calendar week, so the lag is the smallest one consistent with both weeks
    (claim-week Monday minus accident-week Sunday), floored at 0. A day-level reading would
    put the claim before the accident for ~13% of claims (same week, earlier weekday)."""
    a, b = accident_date(c), claim_date(c)
    if not (a and b):
        return None
    a_week_end = a + timedelta(days=6 - a.weekday())
    b_week_start = b - timedelta(days=b.weekday())
    return max(0, (b_week_start - a_week_end).days)
