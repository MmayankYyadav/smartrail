"""
SmartRail - Station Simulation Engine
Coordinates the simulation clock, 8 scheduled trains, delay predictions,
crowd counts, scenario injections, civic discipline, and alert triggers.
Uses fixed random seed (42) for reproducible station states.
"""

import copy
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from delay_model import get_delay_predictor
from allocator import allocate_platforms
from civic import CivicDisciplineManager
from alerts import compute_crowd_risk, get_alert_engine
from announce import generate_coach_guidance_board, build_announcement
import db

RANDOM_SEED = 42

DEFAULT_TRAINS: List[Dict[str, Any]] = [
    {
        "number": "12951",
        "name": "Mumbai Rajdhani",
        "type": "Rajdhani",
        "length": 22,
        "priority": 1,
        "scheduled_arrival": "10:00",
        "usual_platform": "P1",
        "upstream_delay": 0.0,
        "fog_flag": 0,
        "festival_flag": 0,
        "coaches": {
            "A": "EOG + B1-B4 (AC 3-Tier)",
            "B": "H1 (1st AC) + A1-A3 (2nd AC) + PC (Pantry)",
            "C": "B5-B10 (AC 3-Tier) + EOG"
        }
    },
    {
        "number": "12002",
        "name": "Bhopal Shatabdi",
        "type": "Express",
        "length": 18,
        "priority": 1,
        "scheduled_arrival": "10:15",
        "usual_platform": "P2",
        "upstream_delay": 2.0,
        "fog_flag": 0,
        "festival_flag": 0,
        "coaches": {
            "A": "Executive Class (EC1-EC2)",
            "B": "Chair Car (C1-C5)",
            "C": "Chair Car (C6-C12)"
        }
    },
    {
        "number": "12301",
        "name": "Howrah Rajdhani",
        "type": "Rajdhani",
        "length": 24,
        "priority": 1,
        "scheduled_arrival": "10:25",
        "usual_platform": "P1",
        "upstream_delay": 0.0,
        "fog_flag": 0,
        "festival_flag": 0,
        "coaches": {
            "A": "GS / SLR + B1-B4",
            "B": "1A + 2A + Pantry",
            "C": "B5-B11 + SLR"
        }
    },
    {
        "number": "12424",
        "name": "Dibrugarh Rajdhani",
        "type": "Rajdhani",
        "length": 22,
        "priority": 1,
        "scheduled_arrival": "10:40",
        "usual_platform": "P3",
        "upstream_delay": 4.0,
        "fog_flag": 0,
        "festival_flag": 0,
        "coaches": {
            "A": "3A (B1-B4)",
            "B": "1A + 2A",
            "C": "3A (B5-B9)"
        }
    },
    {
        "number": "14217",
        "name": "Prayag Express",
        "type": "Express",
        "length": 24,
        "priority": 2,
        "scheduled_arrival": "10:50",
        "usual_platform": "P1",
        "upstream_delay": 6.0,
        "fog_flag": 0,
        "festival_flag": 1,
        "coaches": {
            "A": "General (GS1-GS3) + S1-S2",
            "B": "AC 2A/3A (A1, B1-B3)",
            "C": "Sleeper (S3-S7) + General (GS4-GS5)"
        }
    },
    {
        "number": "12138",
        "name": "Punjab Mail",
        "type": "Express",
        "length": 24,
        "priority": 2,
        "scheduled_arrival": "11:05",
        "usual_platform": "P3",
        "upstream_delay": 5.0,
        "fog_flag": 0,
        "festival_flag": 0,
        "coaches": {
            "A": "General Unreserved + S1-S3",
            "B": "AC 2A + AC 3A + PC",
            "C": "Sleeper (S4-S8) + General"
        }
    },
    {
        "number": "64491",
        "name": "Ghaziabad Shuttle",
        "type": "Local",
        "length": 12,
        "priority": 3,
        "scheduled_arrival": "11:20",
        "usual_platform": "P2",
        "upstream_delay": 1.0,
        "fog_flag": 0,
        "festival_flag": 0,
        "coaches": {
            "A": "General / Women Coach",
            "B": "General / First Class",
            "C": "General Commuter / Vendor"
        }
    },
    {
        "number": "64492",
        "name": "Palwal Shuttle",
        "type": "Local",
        "length": 12,
        "priority": 3,
        "scheduled_arrival": "11:35",
        "usual_platform": "P2",
        "upstream_delay": 3.0,
        "fog_flag": 0,
        "festival_flag": 0,
        "coaches": {
            "A": "General Coach",
            "B": "General Coach",
            "C": "General Coach"
        }
    }
]

class StationSimulation:
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Base simulation clock: 09:55 AM
        self.start_time = datetime.strptime("09:55", "%H:%M")
        self.current_time = self.start_time
        self.step_count = 0

        # Zone capacities: default A=60, B=40, C=60
        self.zone_capacities = {"A": 60, "B": 40, "C": 60}

        # Train data
        self.trains = copy.deepcopy(DEFAULT_TRAINS)

        # Scenarios / Injected states
        self.blocked_platforms: List[str] = []
        self.surge_platform: Optional[str] = None
        self.surge_zone: Optional[str] = None
        self.injected_train_delays: Dict[str, float] = {}

        # Civic discipline manager
        self.civic_mgr = CivicDisciplineManager(seed=seed)

        # Platform crowd counts: platform -> zone -> count
        self.platforms = ["P1", "P2", "P3"]
        self.zones = ["A", "B", "C"]
        self.crowd_counts: Dict[str, Dict[str, int]] = {
            p: {"A": 22, "B": 12, "C": 25} for p in self.platforms
        }
        self.prev_crowd_counts: Dict[str, Dict[str, int]] = copy.deepcopy(self.crowd_counts)
        self.inflows: Dict[str, Dict[str, float]] = {
            p: {"A": 2.0, "B": 0.8, "C": 2.2} for p in self.platforms
        }

        # Run initial prediction and allocation
        self.update_predictions_and_allocations()

    @property
    def current_time_str(self) -> str:
        return self.current_time.strftime("%H:%M")

    def set_zone_capacities(self, cap_A: int, cap_B: int, cap_C: int):
        self.zone_capacities = {"A": cap_A, "B": cap_B, "C": cap_C}

    def inject_train_delay(self, train_num: str = "12951", delay_mins: float = 12.0):
        """Injects a 12-minute delay on a train."""
        self.injected_train_delays[train_num] = delay_mins
        db.log_event(
            event_type="scenario injection",
            location="Track / Signal",
            details=f"Injected +{delay_mins} min delay on Train {train_num}",
            sim_timestamp=self.current_time_str
        )
        self.update_predictions_and_allocations()

    def toggle_platform_block(self, platform: str = "P2"):
        """Toggles platform blocked / maintenance state."""
        if platform in self.blocked_platforms:
            self.blocked_platforms.remove(platform)
            action = f"Unblocked platform {platform}"
        else:
            self.blocked_platforms.append(platform)
            action = f"Blocked platform {platform} for maintenance"

        db.log_event(
            event_type="platform status",
            location=platform,
            details=action,
            sim_timestamp=self.current_time_str
        )
        self.update_predictions_and_allocations()

    def trigger_crowd_surge(self, platform: str = "P3", zone: str = "C"):
        """Injects a crowd surge into specified platform and zone."""
        self.surge_platform = platform
        self.surge_zone = zone
        self.crowd_counts[platform][zone] = int(self.zone_capacities[zone] * 1.25)
        self.inflows[platform][zone] = 6.8
        
        db.log_event(
            event_type="crowd surge",
            location=f"{platform} Zone {zone}",
            details=f"Crowd surge triggered in {platform} Zone {zone} (Count: {self.crowd_counts[platform][zone]})",
            sim_timestamp=self.current_time_str
        )
        self.update_predictions_and_allocations()

    def reset(self):
        """Resets the station simulation to initial state."""
        self.current_time = self.start_time
        self.step_count = 0
        self.trains = copy.deepcopy(DEFAULT_TRAINS)
        self.blocked_platforms = []
        self.surge_platform = None
        self.surge_zone = None
        self.injected_train_delays = {}
        self.civic_mgr = CivicDisciplineManager(seed=self.seed)
        self.crowd_counts = {p: {"A": 22, "B": 12, "C": 25} for p in self.platforms}
        self.prev_crowd_counts = copy.deepcopy(self.crowd_counts)
        self.inflows = {p: {"A": 2.0, "B": 0.8, "C": 2.2} for p in self.platforms}
        self.update_predictions_and_allocations()
        db.reset_db()
        db.log_event("system", "Station", "Station simulation reset to 09:55 AM", "09:55")

    def update_predictions_and_allocations(self):
        """
        Runs the DelayPredictor and PlatformAllocator across all 8 trains.
        Checks for platform conflicts and creates alerts if necessary.
        """
        predictor = get_delay_predictor()
        alert_engine = get_alert_engine()

        # Step 1: Predict actual delays and arrival times
        for train in self.trains:
            sch_dt = datetime.strptime(train["scheduled_arrival"], "%H:%M")
            hour = sch_dt.hour
            dow = 1 # Tuesday default

            # Get model predicted delay
            base_pred_delay = predictor.predict_delay(
                scheduled_hour=hour,
                day_of_week=dow,
                train_type=train["type"],
                upstream_delay=train["upstream_delay"],
                fog_flag=train["fog_flag"],
                festival_flag=train["festival_flag"]
            )

            # Add any manual what-if injected delay
            injected = self.injected_train_delays.get(train["number"], 0.0)
            final_pred_delay = round(base_pred_delay + injected, 1)

            pred_arr_dt = sch_dt + timedelta(minutes=int(final_pred_delay))
            train["predicted_delay"] = final_pred_delay
            train["predicted_arrival"] = pred_arr_dt.strftime("%H:%M")

        # Step 2: Determine current platform crowd risk profiles
        platform_crowd_risks = {}
        for p in self.platforms:
            max_risk = "Low"
            for z in self.zones:
                c_metric = compute_crowd_risk(
                    zone=z,
                    current_count=self.crowd_counts[p][z],
                    inflow_per_min=self.inflows[p][z],
                    minutes_to_arrival=15,
                    zone_capacity=self.zone_capacities[z]
                )
                if c_metric["risk_level"] == "High":
                    max_risk = "High"
                elif c_metric["risk_level"] == "Medium" and max_risk != "High":
                    max_risk = "Medium"
            platform_crowd_risks[p] = max_risk

        # Step 3: Run rule-based Platform Allocator
        self.trains = allocate_platforms(
            trains=self.trains,
            blocked_platforms=self.blocked_platforms,
            platform_crowd_risks=platform_crowd_risks
        )

        # Step 4: Check if any platform change requires an alert or announcement
        for train in self.trains:
            if train.get("is_platform_changed", False):
                alert_engine.create_platform_conflict_alert(
                    train_num=train["number"],
                    train_name=train["name"],
                    platform=train["recommended_platform"],
                    details=train.get("reason", "Platform reassigned"),
                    sim_time_str=self.current_time_str
                )

    def step(self, minutes: int = 1):
        """Advances the simulation clock by +1 minute and updates crowd/civic states."""
        self.step_count += minutes
        self.current_time += timedelta(minutes=minutes)
        now_str = self.current_time_str
        self.prev_crowd_counts = copy.deepcopy(self.crowd_counts)

        # Dynamic crowd evolution tied to timetable arrivals
        # For each platform, find trains arriving within next 30 minutes
        for p in self.platforms:
            assigned_trains = [
                t for t in self.trains 
                if t.get("recommended_platform") == p and datetime.strptime(t["predicted_arrival"], "%H:%M") >= self.current_time
            ]
            
            if assigned_trains:
                # Find nearest train
                assigned_trains.sort(key=lambda t: datetime.strptime(t["predicted_arrival"], "%H:%M"))
                next_train = assigned_trains[0]
                t_arr = datetime.strptime(next_train["predicted_arrival"], "%H:%M")
                mins_left = max(0, int((t_arr - self.current_time).total_seconds() / 60))

                # Crowd builds up as arrival nears
                urgency = max(0.1, (30 - mins_left) / 30.0)
                
                # Inflow updates
                self.inflows[p]["A"] = round(1.5 + 2.8 * urgency + self.rng.uniform(-0.3, 0.4), 1)
                self.inflows[p]["B"] = round(0.5 + 1.2 * urgency + self.rng.uniform(-0.2, 0.2), 1)
                self.inflows[p]["C"] = round(1.8 + 3.2 * urgency + self.rng.uniform(-0.3, 0.4), 1)

                # Increase counts
                self.crowd_counts[p]["A"] = min(
                    int(self.zone_capacities["A"] * 1.3),
                    int(self.crowd_counts[p]["A"] + self.inflows[p]["A"])
                )
                self.crowd_counts[p]["B"] = min(
                    int(self.zone_capacities["B"] * 1.2),
                    int(self.crowd_counts[p]["B"] + self.inflows[p]["B"])
                )
                self.crowd_counts[p]["C"] = min(
                    int(self.zone_capacities["C"] * 1.4),
                    int(self.crowd_counts[p]["C"] + self.inflows[p]["C"])
                )
            else:
                # No train coming soon -> crowd decays towards baseline
                for z in self.zones:
                    self.inflows[p][z] = 0.5
                    self.crowd_counts[p][z] = max(10, int(self.crowd_counts[p][z] * 0.94))

        # Handle persistent surge if active
        if self.surge_platform and self.surge_zone:
            self.crowd_counts[self.surge_platform][self.surge_zone] = max(
                int(self.zone_capacities[self.surge_zone] * 1.15),
                self.crowd_counts[self.surge_platform][self.surge_zone]
            )

        # Update civic events
        new_civic_events = self.civic_mgr.step_minute(
            sim_time_str=now_str,
            crowd_counts=self.crowd_counts,
            prev_crowd_counts=self.prev_crowd_counts
        )

        # Create alerts for new civic events
        alert_engine = get_alert_engine()
        for c_evt in new_civic_events:
            alert_engine.create_civic_alert(c_evt)

        # Check zone crowd risks and alert if High
        for p in self.platforms:
            for z in self.zones:
                mins_left = 12
                c_metric = compute_crowd_risk(
                    zone=z,
                    current_count=self.crowd_counts[p][z],
                    inflow_per_min=self.inflows[p][z],
                    minutes_to_arrival=mins_left,
                    zone_capacity=self.zone_capacities[z]
                )
                if c_metric["risk_level"] == "High":
                    alert_engine.check_and_create_crowd_alert(
                        platform=p,
                        zone=z,
                        crowd_metric=c_metric,
                        sim_time_str=now_str
                    )

        # Update allocations dynamically
        self.update_predictions_and_allocations()

    def run_guidance_comparison_experiment(
        self,
        share_following_guidance: float = 0.25,
        total_minutes: int = 35
    ) -> Dict[str, Any]:
        """
        Runs the exact same seeded scenario twice:
        1. Without guidance messages
        2. With guidance messages (share_following_guidance moves passengers from overcrowded C to A)
        Returns time-series of peak crowd ratios for plotting.
        """
        rng_no_guide = np.random.RandomState(42)
        rng_with_guide = np.random.RandomState(42)

        times = []
        ratios_no_guide_C = []
        ratios_no_guide_A = []
        ratios_with_guide_C = []
        ratios_with_guide_A = []

        cap_C = self.zone_capacities["C"]
        cap_A = self.zone_capacities["A"]

        # Initial baseline headcounts
        count_C_no = 28
        count_A_no = 24

        count_C_with = 28
        count_A_with = 24

        for m in range(total_minutes):
            t_str = f"T+{m}m"
            times.append(t_str)

            # Inflow curve: rises as train approaches at m=20
            bell_inflow = max(0.5, 5.0 * np.exp(-((m - 20) ** 2) / 45.0))
            
            # 1. No Guidance
            inflow_C_no = bell_inflow * 1.35 + rng_no_guide.normal(0, 0.2)
            inflow_A_no = bell_inflow * 0.65 + rng_no_guide.normal(0, 0.2)
            count_C_no += inflow_C_no
            count_A_no += inflow_A_no

            ratio_C_no = round(count_C_no / cap_C, 2)
            ratio_A_no = round(count_A_no / cap_A, 2)

            ratios_no_guide_C.append(ratio_C_no)
            ratios_no_guide_A.append(ratio_A_no)

            # 2. With Guidance: guidance active starting at m=8 when Zone C passes 0.70 ratio
            if m >= 8:
                diverted = bell_inflow * share_following_guidance * 1.2
                inflow_C_with = (bell_inflow * 1.35) - diverted + rng_with_guide.normal(0, 0.2)
                inflow_A_with = (bell_inflow * 0.65) + diverted + rng_with_guide.normal(0, 0.2)
            else:
                inflow_C_with = bell_inflow * 1.35 + rng_with_guide.normal(0, 0.2)
                inflow_A_with = bell_inflow * 0.65 + rng_with_guide.normal(0, 0.2)

            count_C_with += inflow_C_with
            count_A_with += inflow_A_with

            ratio_C_with = round(count_C_with / cap_C, 2)
            ratio_A_with = round(count_A_with / cap_A, 2)

            ratios_with_guide_C.append(ratio_C_with)
            ratios_with_guide_A.append(ratio_A_with)

        peak_C_no = max(ratios_no_guide_C)
        peak_C_with = max(ratios_with_guide_C)
        peak_reduction_pct = round(((peak_C_no - peak_C_with) / peak_C_no) * 100, 1)

        return {
            "times": times,
            "no_guidance_zone_c": ratios_no_guide_C,
            "no_guidance_zone_a": ratios_no_guide_A,
            "with_guidance_zone_c": ratios_with_guide_C,
            "with_guidance_zone_a": ratios_with_guide_A,
            "peak_no_guidance_C": peak_C_no,
            "peak_with_guidance_C": peak_C_with,
            "peak_reduction_pct": peak_reduction_pct,
            "share_guidance_pct": int(share_following_guidance * 100)
        }

_GLOBAL_SIM = None

def get_station_simulation() -> StationSimulation:
    global _GLOBAL_SIM
    if _GLOBAL_SIM is None:
        _GLOBAL_SIM = StationSimulation(seed=RANDOM_SEED)
    return _GLOBAL_SIM
