"""Volume profile — build volume-at-price histograms for HVN/LVN detection.

Accumulates bar volume into price bins during RTH session. At session reset,
the completed profile is stored. Multiple days' profiles are composited into
a rolling multi-day profile for target selection.

Auction Theory (corrected understanding):
  - HVN (High Volume Node): price ACCEPTED here — buyers and sellers agreed
    on fair value. Acts as MAGNET / TARGET. Price is drawn to HVN because
    responsive participants showed up here before and will again.
  - LVN (Low Volume Node): price REJECTED here — one-time-framing through.
    Acts as HIGHWAY. Price accelerates through LVN (thin structure).
    LVN is NOT a target — it's the fast path BETWEEN targets.
  - Multi-day persistence: A single day's HVN can act as support/resistance
    for 3-5 days. Composite profiles across multiple days capture this.
  - Confluence: When multiple days build HVN at the same price, that
    composite HVN is massively significant (strongest target).
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class VolumeProfile:
    """Accumulates volume-at-price for a single session and classifies HVN/LVN.

    Usage:
        vp = VolumeProfile(bin_size=1.0)
        for bar in bars:
            vp.add_bar(bar["high"], bar["low"], bar["volume"])
        vp.finalize()
    """

    def __init__(self, bin_size: float = 1.0, lvn_threshold: float = 0.5,
                 hvn_threshold: float = 1.5):
        self.bin_size = bin_size
        self.lvn_threshold = lvn_threshold
        self.hvn_threshold = hvn_threshold

        self._bins: Dict[int, float] = defaultdict(float)
        self._finalized = False

        self._mean_vol: float = 0.0
        self._lvn_bins: set = set()
        self._hvn_bins: set = set()
        self._min_bin: int = 0
        self._max_bin: int = 0

    def _price_to_bin(self, price: float) -> int:
        return int(math.floor(price / self.bin_size))

    def _bin_to_price(self, bin_key: int) -> float:
        return bin_key * self.bin_size

    def add_bar(self, high: float, low: float, volume: float) -> None:
        if volume <= 0 or math.isnan(high) or math.isnan(low):
            return
        lo_bin = self._price_to_bin(low)
        hi_bin = self._price_to_bin(high)
        n_bins = hi_bin - lo_bin + 1
        if n_bins <= 0:
            return
        vol_per_bin = volume / n_bins
        for b in range(lo_bin, hi_bin + 1):
            self._bins[b] += vol_per_bin

    def finalize(self) -> None:
        if not self._bins:
            self._finalized = True
            return

        volumes = list(self._bins.values())
        self._mean_vol = sum(volumes) / len(volumes)
        self._min_bin = min(self._bins.keys())
        self._max_bin = max(self._bins.keys())

        lvn_cutoff = self._mean_vol * self.lvn_threshold
        hvn_cutoff = self._mean_vol * self.hvn_threshold

        self._lvn_bins = set()
        self._hvn_bins = set()

        for bin_key, vol in self._bins.items():
            if vol <= lvn_cutoff:
                self._lvn_bins.add(bin_key)
            elif vol >= hvn_cutoff:
                self._hvn_bins.add(bin_key)

        for b in range(self._min_bin, self._max_bin + 1):
            if b not in self._bins:
                self._lvn_bins.add(b)

        self._finalized = True

    def is_empty(self) -> bool:
        return len(self._bins) == 0

    def get_bins(self) -> Dict[int, float]:
        return dict(self._bins)

    def get_poc(self) -> Optional[float]:
        if not self._bins:
            return None
        max_bin = max(self._bins, key=self._bins.get)
        return self._bin_to_price(max_bin) + self.bin_size / 2

    def get_value_area(self, pct: float = 0.70) -> Optional[Tuple[float, float]]:
        if not self._bins:
            return None

        total_vol = sum(self._bins.values())
        target_vol = total_vol * pct

        poc_bin = max(self._bins, key=self._bins.get)
        current_vol = self._bins[poc_bin]
        lo = poc_bin
        hi = poc_bin

        while current_vol < target_vol:
            vol_below = self._bins.get(lo - 1, 0)
            vol_above = self._bins.get(hi + 1, 0)
            if vol_below == 0 and vol_above == 0:
                break
            if vol_below >= vol_above:
                lo -= 1
                current_vol += vol_below
            else:
                hi += 1
                current_vol += vol_above

        val = self._bin_to_price(lo)
        vah = self._bin_to_price(hi) + self.bin_size
        return (val, vah)

    def __repr__(self) -> str:
        n_bins = len(self._bins)
        n_lvn = len(self._lvn_bins) if self._finalized else 0
        n_hvn = len(self._hvn_bins) if self._finalized else 0
        return (f"VolumeProfile(bins={n_bins}, lvn={n_lvn}, hvn={n_hvn}, "
                f"finalized={self._finalized})")


class CompositeProfile:
    """Multi-day composite volume profile for auction theory targeting.

    Merges N days of individual VolumeProfile objects into a single composite.
    The composite reveals persistent HVN areas (where value was accepted across
    multiple sessions) and LVN gaps (where price was consistently rejected).

    Multi-day HVN confluence is the strongest signal: when Friday, Monday, and
    Tuesday all build volume at the same price, that composite HVN is a
    high-probability target.
    """

    def __init__(self, profiles: List[VolumeProfile], bin_size: float = 1.0,
                 lvn_threshold: float = 0.5, hvn_threshold: float = 1.5):
        self.bin_size = bin_size
        self.lvn_threshold = lvn_threshold
        self.hvn_threshold = hvn_threshold

        # Merge all daily profiles into composite bins
        self._bins: Dict[int, float] = defaultdict(float)
        self._day_presence: Dict[int, int] = defaultdict(int)  # how many days had volume at this bin

        for vp in profiles:
            day_bins = vp.get_bins()
            for bin_key, vol in day_bins.items():
                self._bins[bin_key] += vol
                self._day_presence[bin_key] += 1

        self._n_days = len(profiles)
        self._finalized = False

        self._mean_vol: float = 0.0
        self._lvn_bins: set = set()
        self._hvn_bins: set = set()
        self._min_bin: int = 0
        self._max_bin: int = 0

        if self._bins:
            self._finalize()

    def _finalize(self) -> None:
        volumes = list(self._bins.values())
        self._mean_vol = sum(volumes) / len(volumes)
        self._min_bin = min(self._bins.keys())
        self._max_bin = max(self._bins.keys())

        lvn_cutoff = self._mean_vol * self.lvn_threshold
        hvn_cutoff = self._mean_vol * self.hvn_threshold

        self._lvn_bins = set()
        self._hvn_bins = set()

        for bin_key, vol in self._bins.items():
            if vol <= lvn_cutoff:
                self._lvn_bins.add(bin_key)
            elif vol >= hvn_cutoff:
                self._hvn_bins.add(bin_key)

        for b in range(self._min_bin, self._max_bin + 1):
            if b not in self._bins:
                self._lvn_bins.add(b)

        self._finalized = True

    def is_empty(self) -> bool:
        return len(self._bins) == 0

    def _price_to_bin(self, price: float) -> int:
        return int(math.floor(price / self.bin_size))

    def _bin_to_price(self, bin_key: int) -> float:
        return bin_key * self.bin_size

    def find_hvn_below(self, price: float, min_distance: float = 3.0,
                       max_distance: float = 80.0) -> List[Tuple[float, float, int]]:
        """Find HVN zones below the given price (these are TARGETS).

        HVN = where responsive buyers accepted value. Price is drawn back to
        these levels because participants will transact there again.

        Clusters adjacent HVN bins into zones and returns the center of each zone.

        Args:
            price: Current price (entry price)
            min_distance: Minimum distance below price to consider (avoid targeting
                         the price we're already at)
            max_distance: Maximum distance below price to search

        Returns:
            List of (zone_center_price, zone_volume, days_present) sorted by
            distance from price (nearest first). days_present = how many of
            the N composite days had volume at this zone (higher = stronger).
        """
        if not self._finalized or self.is_empty():
            return []

        price_bin = self._price_to_bin(price)
        min_bin = self._price_to_bin(price - max_distance)
        max_bin = self._price_to_bin(price - min_distance)

        # Collect HVN bins in the search range
        hvn_in_range = []
        for b in range(min_bin, max_bin + 1):
            if b in self._hvn_bins:
                hvn_in_range.append(b)

        if not hvn_in_range:
            return []

        # Cluster adjacent HVN bins into zones
        hvn_in_range.sort()
        zones = []
        current_zone = [hvn_in_range[0]]

        for i in range(1, len(hvn_in_range)):
            if hvn_in_range[i] - hvn_in_range[i - 1] <= 2:  # Adjacent or 1 gap
                current_zone.append(hvn_in_range[i])
            else:
                zones.append(current_zone)
                current_zone = [hvn_in_range[i]]
        zones.append(current_zone)

        # For each zone: compute center, total volume, days present
        results = []
        for zone_bins in zones:
            total_vol = sum(self._bins.get(b, 0) for b in zone_bins)
            # Volume-weighted center
            if total_vol > 0:
                center = sum(self._bin_to_price(b) * self._bins.get(b, 0)
                             for b in zone_bins) / total_vol
                center += self.bin_size / 2  # bin midpoint
            else:
                center = self._bin_to_price(zone_bins[len(zone_bins) // 2]) + self.bin_size / 2

            # How many days had volume at any bin in this zone
            days = max(self._day_presence.get(b, 0) for b in zone_bins)

            results.append((center, total_vol, days))

        # Sort by distance from entry (nearest HVN first)
        results.sort(key=lambda x: abs(price - x[0]))
        return results

    def find_hvn_above(self, price: float, min_distance: float = 3.0,
                       max_distance: float = 80.0) -> List[Tuple[float, float, int]]:
        """Find HVN zones above the given price (targets for LONG trades).

        Mirror of find_hvn_below for long-side targeting.

        Returns:
            List of (zone_center_price, zone_volume, days_present) sorted by
            distance from price (nearest first).
        """
        if not self._finalized or self.is_empty():
            return []

        min_bin = self._price_to_bin(price + min_distance)
        max_bin = self._price_to_bin(price + max_distance)

        hvn_in_range = []
        for b in range(min_bin, max_bin + 1):
            if b in self._hvn_bins:
                hvn_in_range.append(b)

        if not hvn_in_range:
            return []

        hvn_in_range.sort()
        zones = []
        current_zone = [hvn_in_range[0]]

        for i in range(1, len(hvn_in_range)):
            if hvn_in_range[i] - hvn_in_range[i - 1] <= 2:
                current_zone.append(hvn_in_range[i])
            else:
                zones.append(current_zone)
                current_zone = [hvn_in_range[i]]
        zones.append(current_zone)

        results = []
        for zone_bins in zones:
            total_vol = sum(self._bins.get(b, 0) for b in zone_bins)
            if total_vol > 0:
                center = sum(self._bin_to_price(b) * self._bins.get(b, 0)
                             for b in zone_bins) / total_vol
                center += self.bin_size / 2
            else:
                center = self._bin_to_price(zone_bins[len(zone_bins) // 2]) + self.bin_size / 2

            days = max(self._day_presence.get(b, 0) for b in zone_bins)
            results.append((center, total_vol, days))

        results.sort(key=lambda x: abs(price - x[0]))
        return results

    def path_has_lvn(self, entry: float, target: float) -> float:
        """Check how much of the path between entry and target is through LVN (thin structure).

        Returns fraction in [0, 1]:
          0.0 = entire path is through thick structure — price will stall
          1.0 = entire path is through air — price will accelerate

        This is a BONUS metric, not the primary targeting signal.
        LVN in the path means faster fill speed, not a better target.
        """
        if not self._finalized or self.is_empty():
            return 0.5  # neutral if no data

        price_lo = min(entry, target)
        price_hi = max(entry, target)
        lo_bin = self._price_to_bin(price_lo)
        hi_bin = self._price_to_bin(price_hi)
        n_bins = hi_bin - lo_bin + 1
        if n_bins <= 0:
            return 0.5

        lvn_count = sum(1 for b in range(lo_bin, hi_bin + 1)
                        if b in self._lvn_bins or b not in self._bins)
        return lvn_count / n_bins

    def is_hvn_at(self, price: float, tolerance: float = 2.0) -> bool:
        """Check if there's an HVN at or very near the given price.

        Used to check whether a support level coincides with a composite HVN
        (confluence = strongest possible target).
        """
        if not self._finalized or self.is_empty():
            return False

        center_bin = self._price_to_bin(price)
        tol_bins = max(1, int(tolerance / self.bin_size))

        for b in range(center_bin - tol_bins, center_bin + tol_bins + 1):
            if b in self._hvn_bins:
                return True
        return False

    def get_composite_poc(self) -> Optional[float]:
        """Return composite Point of Control (highest volume price across all days)."""
        if not self._bins:
            return None
        max_bin = max(self._bins, key=self._bins.get)
        return self._bin_to_price(max_bin) + self.bin_size / 2

    def __repr__(self) -> str:
        n_bins = len(self._bins)
        n_lvn = len(self._lvn_bins) if self._finalized else 0
        n_hvn = len(self._hvn_bins) if self._finalized else 0
        return (f"CompositeProfile(days={self._n_days}, bins={n_bins}, "
                f"lvn={n_lvn}, hvn={n_hvn})")
