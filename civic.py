"""
SmartRail - Civic Discipline Module
Simulates civic events (littering, yellow-line crossings, dustbin fill, queue congestion).
Maintains cleaning priority ranking and zone-wise Discipline Scores (0-100).
"""

import random
from typing import Dict, List, Any, Tuple

RANDOM_SEED = 42

class CivicDisciplineManager:
    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)
        
        # Dustbin fill % per platform (starts between 30% and 55%)
        self.dustbin_fill: Dict[str, float] = {
            "P1": 42.0,
            "P2": 35.0,
            "P3": 68.0
        }

        # Track active event counts per (platform, zone)
        # Zones are (P1, A), (P1, B), (P1, C), (P2, A), etc.
        self.platforms = ["P1", "P2", "P3"]
        self.zones = ["A", "B", "C"]
        
        self.littering_counts: Dict[str, int] = {f"{p}-{z}": 0 for p in self.platforms for z in self.zones}
        self.yellow_line_counts: Dict[str, int] = {f"{p}-{z}": 0 for p in self.platforms for z in self.zones}
        self.queue_congestion_counts: Dict[str, int] = {f"{p}-{z}": 0 for p in self.platforms for z in self.zones}

        # History of recent incidents
        self.recent_events: List[Dict[str, Any]] = []

    def step_minute(
        self,
        sim_time_str: str,
        crowd_counts: Dict[str, Dict[str, int]],
        prev_crowd_counts: Dict[str, Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        """
        Advances civic simulation by 1 minute:
        - Dustbins gradually fill (faster on busier platforms)
        - Probabilistic littering and yellow-line crossings based on crowd density
        - Detects sharp jumps in crowd -> queue congestion
        Returns a list of newly triggered civic events for alerting.
        """
        new_events = []

        # 1. Update dustbin fills
        for p in self.platforms:
            fill_increment = self.rng.uniform(0.6, 2.2)
            self.dustbin_fill[p] = min(100.0, round(self.dustbin_fill[p] + fill_increment, 1))
            
            # Check for dustbin full alert threshold (>= 85%)
            if self.dustbin_fill[p] >= 85.0:
                event = {
                    "event_type": "dustbin full",
                    "location": f"{p}",
                    "details": f"Dustbin fill reached {self.dustbin_fill[p]:.1f}% (Threshold: 85%)",
                    "timestamp": sim_time_str,
                    "risk_level": "High" if self.dustbin_fill[p] >= 90 else "Medium",
                    "recommended_action": f"Dispatch sanitation team to empty dustbin on {p}"
                }
                new_events.append(event)

        # 2. Stochastic civic events per zone
        for p in self.platforms:
            for z in self.zones:
                pz_key = f"{p}-{z}"
                count = crowd_counts.get(p, {}).get(z, 20)
                prev_count = prev_crowd_counts.get(p, {}).get(z, count)

                # Queue congestion: sharp jump in crowd (>= 12 persons in 1 min)
                if (count - prev_count) >= 12 or (count > 50 and self.rng.random() < 0.12):
                    self.queue_congestion_counts[pz_key] += 1
                    event = {
                        "event_type": "queue congestion",
                        "location": f"{p} Zone {z}",
                        "details": f"Sudden crowd surge (+{count - prev_count} pax) in {p} Zone {z}. Current: {count}",
                        "timestamp": sim_time_str,
                        "risk_level": "High" if count > 55 else "Medium",
                        "recommended_action": f"Deploy queue marshall to organize boarding line at {p} Zone {z}"
                    }
                    new_events.append(event)
                    self.recent_events.append(event)

                # Littering event probability (higher in general coaches A & C)
                litter_prob = 0.08 if z in ("A", "C") else 0.03
                if count > 45:
                    litter_prob *= 1.8
                if self.rng.random() < litter_prob:
                    self.littering_counts[pz_key] += 1
                    event = {
                        "event_type": "littering",
                        "location": f"{p} Zone {z}",
                        "details": f"Littering detected near platform seating in {p} Zone {z}",
                        "timestamp": sim_time_str,
                        "risk_level": "Low",
                        "recommended_action": f"Send cleaning sweep to {p} Zone {z}"
                    }
                    new_events.append(event)
                    self.recent_events.append(event)

                # Yellow-line crossing event (safety risk, higher with large crowd)
                yellow_line_prob = 0.06 if count > 40 else 0.02
                if self.rng.random() < yellow_line_prob:
                    self.yellow_line_counts[pz_key] += 1
                    event = {
                        "event_type": "yellow-line",
                        "location": f"{p} Zone {z}",
                        "details": f"Passenger stepped over the yellow safety line at {p} Zone {z}",
                        "timestamp": sim_time_str,
                        "risk_level": "High",
                        "recommended_action": f"Sound yellow-line audio warning on {p} Zone {z}"
                    }
                    new_events.append(event)
                    self.recent_events.append(event)

        # Keep recent events history bounded
        if len(self.recent_events) > 50:
            self.recent_events = self.recent_events[-50:]

        return new_events

    def get_discipline_score(self, platform: str, zone: str) -> Tuple[int, str]:
        """
        Discipline Score per zone, 0-100:
        Start at 100
        Subtract 8 per littering event
        Subtract 15 per yellow-line event
        Subtract 10 per queue congestion event
        Subtract 5 if dustbin >= 85%
        Clamp to 0-100.
        Returns: (score, color_category)
        """
        score = 100
        pz_key = f"{platform}-{zone}"

        score -= self.littering_counts.get(pz_key, 0) * 8
        score -= self.yellow_line_counts.get(pz_key, 0) * 15
        score -= self.queue_congestion_counts.get(pz_key, 0) * 10
        if self.dustbin_fill.get(platform, 0.0) >= 85.0:
            score -= 5

        score = max(0, min(100, score))

        if score >= 80:
            color = "green" # Good
        elif score >= 50:
            color = "amber" # Moderate
        else:
            color = "red"   # Critical

        return score, color

    def get_cleaning_priority_list(self) -> List[Dict[str, Any]]:
        """
        Cleaning priority list sorted by dustbin fill and littering count descending.
        """
        priority_rows = []
        for p in self.platforms:
            bin_fill = self.dustbin_fill[p]
            for z in self.zones:
                pz_key = f"{p}-{z}"
                litter = self.littering_counts[pz_key]
                # Priority index: dustbin weight 0.6 + littering count * 8
                urgency = round(bin_fill * 0.5 + litter * 10.0, 1)
                
                status = "Clean"
                if urgency > 60 or bin_fill >= 85 or litter >= 2:
                    status = "Urgent Clean Needed"
                elif urgency > 35 or litter >= 1:
                    status = "Routine Sweep"

                priority_rows.append({
                    "platform": p,
                    "zone": z,
                    "location": f"{p} - Zone {z}",
                    "dustbin_fill_pct": bin_fill,
                    "litter_count": litter,
                    "urgency_score": urgency,
                    "status": status
                })

        priority_rows.sort(key=lambda x: (x["dustbin_fill_pct"] >= 85, x["urgency_score"]), reverse=True)
        return priority_rows

    def execute_cleaning_task(self, platform: str, zone: Optional[str] = None):
        """Operator approved cleaning task -> resets litter count & empties dustbin."""
        if zone:
            pz_key = f"{platform}-{zone}"
            self.littering_counts[pz_key] = 0
            self.queue_congestion_counts[pz_key] = max(0, self.queue_congestion_counts[pz_key] - 1)
        # Empty dustbin on that platform
        self.dustbin_fill[platform] = max(10.0, self.dustbin_fill[platform] - 60.0)

    def trigger_litter_event(self, platform: str = "P3", zone: str = "C"):
        """Manual scenario injection for demo."""
        pz_key = f"{platform}-{zone}"
        self.littering_counts[pz_key] += 1
        self.dustbin_fill[platform] = max(self.dustbin_fill[platform], 88.0)
