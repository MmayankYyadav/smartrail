"""
SmartRail - Platform Allocator Module
Rule-based scoring system for assigning platforms to arriving trains.
Considers:
1. Availability at predicted arrival + dwell buffer (5 min)
2. Train length vs platform capacity (P1:24, P2:22, P3:24)
3. Train priority resolution (1 > 2 > 3)
4. Platform crowd risk avoidance (avoid High risk)
5. Preservation of usual/scheduled platform to minimize passenger confusion
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Platform coach lengths
PLATFORM_LENGTHS = {
    "P1": 24,
    "P2": 22,
    "P3": 24
}

# Dwell times in minutes by train type
DWELL_TIMES = {
    "Rajdhani": 10,
    "Express": 8,
    "Local": 5
}
BUFFER_MINUTES = 5

def parse_time(t_str: str) -> datetime:
    """Parses HH:MM into a dummy datetime for time math."""
    return datetime.strptime(t_str, "%H:%M")

def format_time(dt: datetime) -> str:
    """Formats datetime to HH:MM."""
    return dt.strftime("%H:%M")

def check_overlap(start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> bool:
    """Returns True if two time intervals overlap (strictly overlap or buffer overlap)."""
    return max(start1, start2) < min(end1, end2)

def evaluate_platform_score(
    platform: str,
    train: Dict[str, Any],
    pred_arr_dt: datetime,
    dwell_mins: int,
    blocked_platforms: List[str],
    platform_schedule: Dict[str, List[Dict[str, Any]]],
    platform_crowd_risks: Dict[str, str],
    current_train_num: str
) -> Tuple[float, List[str], bool]:
    """
    Computes a score for a candidate platform.
    Returns: (score, list_of_reasons, is_feasible)
    """
    train_len = train.get("length", 24)
    usual_plat = train.get("usual_platform", "P1")
    train_priority = train.get("priority", 2)
    pred_dep_dt = pred_arr_dt + timedelta(minutes=dwell_mins + BUFFER_MINUTES)

    reasons = []

    # Hard Constraint 1: Platform Blocked (e.g. maintenance)
    if platform in blocked_platforms:
        return -9999.0, [f"{platform} is BLOCKED for maintenance"], False

    # Hard Constraint 2: Train length fits platform length
    max_len = PLATFORM_LENGTHS[platform]
    if train_len > max_len:
        return -9999.0, [f"Length {train_len} coaches exceeds {platform} limit ({max_len})"], False

    # Hard Constraint 3: Time availability & overlap check with already committed trains
    conflicts = []
    for occ in platform_schedule.get(platform, []):
        if occ["train_number"] == current_train_num:
            continue
        occ_start = occ["start"]
        occ_end = occ["end"]
        if check_overlap(pred_arr_dt, pred_dep_dt, occ_start, occ_end):
            conflicts.append(occ)

    if conflicts:
        # Check if current train has strictly higher priority than all conflicting trains
        all_conflicts_lower_priority = all(train_priority < c["priority"] for c in conflicts)
        if not all_conflicts_lower_priority:
            busy_reasons = [f"busy with {c['train_name']} until {format_time(c['end'])}" for c in conflicts]
            return -9999.0, [f"{platform} " + ", ".join(busy_reasons)], False
        else:
            reasons.append(f"Priority {train_priority} overrides conflicting lower-priority trains")

    # Soft Scoring
    score = 100.0

    # Preference for scheduled / usual platform
    if platform == usual_plat:
        score += 25.0
        reasons.append("Matches scheduled platform")
    else:
        score -= 15.0
        reasons.append(f"Changed from scheduled {usual_plat}")

    # Crowd risk adjustment
    risk = platform_crowd_risks.get(platform, "Low")
    if risk == "Low":
        score += 15.0
        reasons.append("Crowd risk Low")
    elif risk == "Medium":
        score += 0.0
        reasons.append("Crowd risk Medium")
    elif risk == "High":
        score -= 40.0
        reasons.append("Crowd risk High (penalty applied)")

    # Length fit optimality: prefer snug fit to leave bigger platforms open if possible
    # But P1 and P3 are 24, P2 is 22
    if train_len <= 22 and platform == "P2":
        score += 5.0
        reasons.append("Optimal coach capacity fit")

    return score, reasons, True

def allocate_platforms(
    trains: List[Dict[str, Any]],
    blocked_platforms: Optional[List[str]] = None,
    platform_crowd_risks: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Takes the list of simulated trains (with predicted_delay and predicted_arrival),
    and assigns the optimal platform for each train.
    
    Sorts allocation order by Priority (1 first), then by Predicted Arrival time.
    """
    if blocked_platforms is None:
        blocked_platforms = []
    if platform_crowd_risks is None:
        platform_crowd_risks = {"P1": "Low", "P2": "Low", "P3": "Low"}

    # Track allocated time intervals per platform: list of dicts {train_number, train_name, priority, start, end}
    platform_schedule: Dict[str, List[Dict[str, Any]]] = {
        "P1": [],
        "P2": [],
        "P3": []
    }

    # Sort trains for allocation by priority (ascending: 1 is top), then predicted arrival
    def get_sort_key(t):
        arr_time = parse_time(t["predicted_arrival"])
        return (t.get("priority", 2), arr_time)

    sorted_trains = sorted(trains, key=get_sort_key)
    results_map = {}

    for train in sorted_trains:
        t_num = train["number"]
        t_name = train["name"]
        t_type = train.get("type", "Express")
        t_len = train.get("length", 24)
        t_prio = train.get("priority", 2)
        usual_plat = train.get("usual_platform", "P1")
        dwell_mins = DWELL_TIMES.get(t_type, 8)

        pred_arr_dt = parse_time(train["predicted_arrival"])
        pred_dep_dt = pred_arr_dt + timedelta(minutes=dwell_mins + BUFFER_MINUTES)

        best_platform = None
        best_score = -float("inf")
        best_reasons = []
        platform_evaluations = {}

        candidate_platforms = ["P1", "P2", "P3"]

        # If usual platform is feasible and free, evaluate it first
        for plat in candidate_platforms:
            score, reasons, is_feasible = evaluate_platform_score(
                platform=plat,
                train=train,
                pred_arr_dt=pred_arr_dt,
                dwell_mins=dwell_mins,
                blocked_platforms=blocked_platforms,
                platform_schedule=platform_schedule,
                platform_crowd_risks=platform_crowd_risks,
                current_train_num=t_num
            )
            platform_evaluations[plat] = (score, reasons, is_feasible)

            if is_feasible and score > best_score:
                best_score = score
                best_platform = plat
                best_reasons = reasons

        # Fallback if all platforms have conflicts or blocked:
        if best_platform is None:
            # Fall back to usual platform or first non-blocked platform that fits length
            for plat in candidate_platforms:
                if plat not in blocked_platforms and t_len <= PLATFORM_LENGTHS[plat]:
                    best_platform = plat
                    best_reasons = [f"Emergency fallback: only length-compatible platform available"]
                    break
            if best_platform is None:
                best_platform = usual_plat
                best_reasons = ["Fallback to timetable platform despite constraints"]

        # Commit schedule for this train
        platform_schedule[best_platform].append({
            "train_number": t_num,
            "train_name": t_name,
            "priority": t_prio,
            "start": pred_arr_dt,
            "end": pred_dep_dt
        })

        is_changed = (best_platform != usual_plat)
        
        # Build plain language reason string
        # e.g. "P3 chosen: P2 busy until 10:14, length fits (24 coaches), crowd risk Medium."
        if is_changed:
            primary_reason = f"{best_platform} chosen: Scheduled {usual_plat} "
            if usual_plat in blocked_platforms:
                primary_reason += "blocked for maintenance; "
            elif t_len > PLATFORM_LENGTHS[usual_plat]:
                primary_reason += f"cannot fit length {t_len}; "
            elif platform_crowd_risks.get(usual_plat) == "High":
                primary_reason += "has High crowd congestion; "
            else:
                primary_reason += "has train occupancy conflict; "
            
            fit_note = f"length fits ({t_len} coaches)"
            crowd_note = f"crowd risk {platform_crowd_risks.get(best_platform, 'Low')}"
            final_reason = f"{primary_reason}{fit_note}, {crowd_note}."
        else:
            final_reason = f"{best_platform} chosen: Scheduled platform free until {format_time(pred_dep_dt)}, length fits ({t_len} coaches), crowd risk {platform_crowd_risks.get(best_platform, 'Low')}."

        results_map[t_num] = {
            "recommended_platform": best_platform,
            "is_platform_changed": is_changed,
            "reason": final_reason,
            "dwell_mins": dwell_mins,
            "occupancy_start": format_time(pred_arr_dt),
            "occupancy_end": format_time(pred_dep_dt),
            "eval_notes": best_reasons
        }

    # Return updated trains preserving original order
    updated_trains = []
    for train in trains:
        t_copy = dict(train)
        alloc_data = results_map.get(train["number"], {})
        t_copy.update(alloc_data)
        updated_trains.append(t_copy)

    return updated_trains
