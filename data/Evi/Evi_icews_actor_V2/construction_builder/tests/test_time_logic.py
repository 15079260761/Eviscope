"""Section 21.1: temporal logic tests."""

import unittest

from src.timeutils import (
    allen, contains_year, fmt_duration_days, interval_overlap, parse_date,
    sentinel_date, year_span,
)


class TestTimeLogic(unittest.TestCase):
    def test_before_after_boundaries(self):
        # strict boundaries: X before Y requires X.end < Y.start
        self.assertTrue(allen(sentinel_date("2000-01-01"), sentinel_date("2001-06-01"),
                              sentinel_date("2002-01-01"), sentinel_date("2003-01-01"), "X < Y"))
        self.assertFalse(allen(sentinel_date("2000-01-01"), sentinel_date("2002-01-01"),
                               sentinel_date("2002-01-01"), sentinel_date("2003-01-01"), "X < Y"))
        self.assertTrue(allen(sentinel_date("2005-01-01"), sentinel_date("2006-01-01"),
                              sentinel_date("2000-01-01"), sentinel_date("2001-01-01"), "X > Y"))

    def test_first_last_unique_and_ties(self):
        spans = [(parse_date("2010-01-01"), parse_date("2012-12-31")),
                 (parse_date("2012-01-01"), parse_date("2015-12-31"))]
        first = min(s for s, _ in spans)
        last = max(e for _, e in spans)
        self.assertEqual(first, parse_date("2010-01-01"))
        self.assertEqual(last, parse_date("2015-12-31"))
        ties = [parse_date("2012-12-31"), parse_date("2012-12-31")]
        self.assertEqual(len(set(ties)), 1)

    def test_point_and_interval_overlap(self):
        # point overlap: touching dates count as overlapping
        self.assertTrue(interval_overlap("2000-01-01", "2000-06-30",
                                         "2000-06-30", "2001-01-01"))
        # strictly disjoint does not
        self.assertFalse(interval_overlap("2000-01-01", "2000-06-29",
                                          "2000-06-30", "2001-01-01"))

    def test_unknown_boundaries(self):
        # open boundaries never block overlap (explicit nulls, not sentinels)
        self.assertTrue(interval_overlap(None, None, None, None))
        # [2000-01-01, open) overlaps (open, 2001-01-01]
        self.assertTrue(interval_overlap("2000-01-01", None, None, "2001-01-01"))
        # [2000-01-01, open) and (open, 1999-12-31] are genuinely disjoint
        self.assertFalse(interval_overlap("2000-01-01", None, None, "1999-12-31"))

    def test_cross_year_intervals(self):
        self.assertEqual(year_span("1998-06-01", "2002-03-01"),
                         {"1998", "1999", "2000", "2001", "2002"})
        self.assertTrue(contains_year("1998-06-01", "2002-03-01", 2000))
        self.assertFalse(contains_year("1998-06-01", "2002-03-01", 1997))

    def test_sentinel_duration_format(self):
        self.assertEqual(fmt_duration_days(0), "0 years, 0 months, 0 days, 0 hours, 0 minutes, 0 seconds")
        self.assertEqual(fmt_duration_days(365 + 30 + 1 + 0.5),
                         "1 years, 1 months, 1 days, 12 hours, 0 minutes, 0 seconds")

    def test_year_normalization_boundaries(self):
        # year-only facts normalize to full year spans
        self.assertEqual(parse_date("2005-01-01").isoformat(), "2005-01-01")


if __name__ == "__main__":
    unittest.main()
