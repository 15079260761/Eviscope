"""Temporal primitives shared by both sub-benchmarks.

One time module for all comparisons (section 6.3 of the construction
document).  Open bounds are explicit values, never arbitrary sentinel years:

  * ICEWS  uses 'beginning of time' / 'end of time' with sentinels
    0001-01-01 / 9999-12-31 only where the source generator itself used them
    for interval arithmetic (verified against all 89,372 source answers).
  * CronQuestions uses year-granularity intervals; year-only facts are
    normalized to [YYYY-01-01, YYYY-12-31] by the normalizer.

Time comparisons in *this* module work on ISO date strings with '' as the
open bound, plus the sentinel constants below for the ICEWS answer formats.
"""

from __future__ import annotations

from datetime import date, timedelta

BOT = "beginning of time"
EOT = "end of time"
BOT_DATE = date(1, 1, 1)
EOT_DATE = date(9999, 12, 31)
MAXSPAN_DAYS = (EOT_DATE - BOT_DATE).days
#: Sum/average durations beyond 2000 mean-Gregorian years are reported as
#: 'forever' by the source generator (verified empirically on all rows).
FOREVER_THRESHOLD = 2000 * 365.2425


def parse_date(value: str):
    """Parse an ISO date; ''/None means open.  Returns date or ''."""
    if value in (None, ""):
        return ""
    if value == BOT:
        return BOT_DATE
    if value == EOT:
        return EOT_DATE
    return date.fromisoformat(value)


def sentinel_date(value: str) -> date:
    """Map open-bound words to the generator's arithmetic sentinels."""
    if value == BOT:
        return BOT_DATE
    if value == EOT:
        return EOT_DATE
    return date.fromisoformat(value)


def interval_overlap(a_start, a_end, b_start, b_end) -> bool:
    """Open boundaries are open, not arbitrary sentinel years."""
    a_s, a_e = parse_date(a_start), parse_date(a_end)
    b_s, b_e = parse_date(b_start), parse_date(b_end)
    return (a_e == "" or b_s == "" or a_e >= b_s) and (b_e == "" or a_s == "" or b_e >= a_s)


def year_span(start, end, max_years: int = 500) -> set[str]:
    s, e = parse_date(start), parse_date(end)
    if s == "" or e == "":
        return set()
    first, last = s.year, e.year
    if last - first > max_years:
        return set()
    return {str(y) for y in range(first, last + 1)}


def contains_year(start, end, year: int) -> bool:
    s, e = parse_date(start), parse_date(end)
    if s == "" or e == "":
        return False
    return s.year <= year <= e.year


# --------------------------------------------------------------------------- #
# Allen interval relations (used by the ICEWS programs)
# --------------------------------------------------------------------------- #

def allen(x_lo, x_hi, y_lo, y_hi, code: str):
    """Evaluate an Allen-style relation code between intervals x and y.

    Bounds are date objects or '' for open.  Open bounds compare naturally:
    '' sorts below everything when used as a lower bound is handled by the
    caller (the ICEWS programs always pass concrete sentinel dates).
    """
    if code == "X = Y":
        return x_lo == y_lo and x_hi == y_hi
    if code == "X < Y":
        return x_hi < y_lo
    if code == "X > Y":
        return y_hi < x_lo
    if code == "X d Y":   # X during Y
        return x_lo >= y_lo and x_hi <= y_hi
    if code == "X di Y":  # Y during X
        return y_lo >= x_lo and y_hi <= x_hi
    if code == "X s Y":   # X starts Y
        return x_lo == y_lo and x_hi < y_hi
    if code == "X si Y":
        return y_lo == x_lo and y_hi < x_hi
    if code == "X f Y":   # X finishes Y
        return x_hi == y_hi and x_lo > y_lo
    if code == "X fi Y":
        return y_hi == x_hi and y_lo > x_lo
    if code == "X o Y":   # X overlaps Y
        return x_lo < y_lo < x_hi < y_hi
    if code == "X oi Y":
        return y_lo < x_lo < y_hi < x_hi
    if code == "X m Y":   # X meets Y
        return x_hi == y_lo
    if code == "X mi Y":
        return y_hi == x_lo
    return None


# --------------------------------------------------------------------------- #
# ICEWS answer formatting (formats verified against every source answer)
# --------------------------------------------------------------------------- #

def fmt_interval_bound(bound, is_start: bool) -> str:
    if bound == BOT_DATE:
        return "0001-01-01T00:00:00.000000"
    if bound == EOT_DATE:
        return "9999-12-31T23:59:59.999999"
    return bound.isoformat()


def fmt_interval(lo, hi) -> str:
    """Interval tuple used by union/intersection answers; lo/hi are date
    objects (possibly sentinels)."""
    return f"({fmt_interval_bound(lo, True)}, {fmt_interval_bound(hi, False)})"


def fmt_duration_days(total_days: float) -> str:
    """'N years, M months, D days, H hours, 0 minutes, 0 seconds' with the
    source generator's 365-day year / 30-day month decomposition."""
    y = int(total_days // 365)
    rem = total_days - y * 365
    m = int(rem // 30)
    rem -= m * 30
    d = int(rem)
    frac = rem - d
    h = int(frac * 24 + 1e-9)
    return f"{y} years, {m} months, {d} days, {h} hours, 0 minutes, 0 seconds"


def duration_days(start: str, end: str) -> int:
    """Duration in days with the source generator's sentinel arithmetic."""
    return (sentinel_date(end) - sentinel_date(start)).days
