"""
SmartRail - Crowd Risk & Alert Engine Module
Evaluates zone crowd risk:
  predicted_ratio = (current_count + inflow_per_min * minutes_to_arrival) / zone_capacity
Generates HEADLINE YES/NO actions: "Send staff to Platform X Zone Y? YES/NO".
Manages human-in-the-loop alert lifecycle (Approve, Reject, Override, Mark completed).
"""

import uuid
from typing import Dict, List, Any, Tuple, Optional
import db

def compute_crowd_risk(
    zone: str,
    current_count: int,
    inflow_per_min: float,
    minutes_to_arrival: int,
    zone_capacity: int
) -> Dict[str, Any]:
    """
    Computes crowd metrics:
    - predicted_ratio = (current_count + inflow_per_min * minutes_to_arrival) / zone_capacity
    - Risk category:
        General (A, C): Low < 0.5, Medium 0.5-0.8, High >= 0.8
        Zone B (AC/Sleeper): Low < 0.7, Medium 0.7-1.0, High >= 1.0
    - Risk score: 0-100
    - Estimated clearance time: minutes needed to clear queue at 15 pax/min
    - HEADLINE ACTION: "Send staff to Platform X Zone Y? YES" (High) or "NO" (Low/Med)
    """
    safe_cap = max(1, zone_capacity)
    predicted_headcount = current_count + (inflow_per_min * max(0, minutes_to_arrival))
    predicted_ratio = round(predicted_headcount / safe_cap, 2)

    # Risk thresholds
    if zone in ("A", "C"):
        if predicted_ratio < 0.5:
            risk_level = "Low"
        elif predicted_ratio < 0.8:
            risk_level = "Medium"
        else:
            risk_level = "High"
    else: # Zone B
        if predicted_ratio < 0.7:
            risk_level = "Low"
        elif predicted_ratio < 1.0:
            risk_level = "Medium"
        else:
            risk_level = "High"

    risk_score = min(100, int(predicted_ratio * 100))
    # Clearance time assuming 15 passengers processed/boarded per minute
    clearance_time_mins = max(1, int(current_count / 15))

    staff_action_yes_no = "YES" if risk_level == "High" else "NO"

    return {
        "zone": zone,
        "current_count": current_count,
        "inflow_per_min": inflow_per_min,
        "minutes_to_arrival": minutes_to_arrival,
        "zone_capacity": zone_capacity,
        "predicted_headcount": int(predicted_headcount),
        "predicted_ratio": predicted_ratio,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "clearance_time_mins": clearance_time_mins,
        "staff_action_yes_no": staff_action_yes_no,
        "headline_action": f"Send staff to {zone}? {staff_action_yes_no}"
    }

class AlertEngine:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or db.DEFAULT_DB_PATH
        db.init_db(self.db_path)

    def generate_alert_id(self, prefix: str = "ALT") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:6].upper()}"

    def check_and_create_crowd_alert(
        self,
        platform: str,
        zone: str,
        crowd_metric: Dict[str, Any],
        sim_time_str: str
    ) -> Optional[Dict[str, Any]]:
        """Generates an alert if crowd risk is High."""
        if crowd_metric["risk_level"] == "High":
            alert_id = f"CRW-{platform}{zone}-{sim_time_str.replace(':', '')}"
            location = f"{platform} Zone {zone}"
            recommended_action = f"Deploy crowd regulation marshals to {platform} Zone {zone}. Divert incoming general passengers."
            confidence = round(min(0.98, 0.85 + (crowd_metric["predicted_ratio"] - 0.8) * 0.2), 2)
            
            created = db.log_alert(
                alert_id=alert_id,
                alert_type="crowd risk",
                location=location,
                sim_timestamp=sim_time_str,
                risk_level="High",
                recommended_action=recommended_action,
                confidence=confidence,
                db_path=self.db_path
            )
            if created:
                db.log_event(
                    event_type="crowd risk alert created",
                    location=location,
                    details=f"High crowd risk detected (Ratio {crowd_metric['predicted_ratio']:.2f}, Score {crowd_metric['risk_score']})",
                    sim_timestamp=sim_time_str,
                    db_path=self.db_path
                )
            return {
                "id": alert_id,
                "type": "crowd risk",
                "location": location,
                "timestamp": sim_time_str,
                "risk_level": "High",
                "recommended_action": recommended_action,
                "confidence": confidence,
                "status": "pending"
            }
        return None

    def create_platform_conflict_alert(
        self,
        train_num: str,
        train_name: str,
        platform: str,
        details: str,
        sim_time_str: str
    ) -> Dict[str, Any]:
        """Logs a platform conflict alert."""
        alert_id = f"PLT-{train_num}-{sim_time_str.replace(':', '')}"
        location = f"{platform}"
        rec_action = f"Approve reassignment for {train_name} ({train_num}) to alternative platform."
        
        db.log_alert(
            alert_id=alert_id,
            alert_type="platform conflict",
            location=location,
            sim_timestamp=sim_time_str,
            risk_level="High",
            recommended_action=rec_action,
            confidence=0.96,
            db_path=self.db_path
        )
        db.log_event(
            event_type="platform conflict",
            location=location,
            details=details,
            sim_timestamp=sim_time_str,
            db_path=self.db_path
        )
        return {
            "id": alert_id,
            "type": "platform conflict",
            "location": location,
            "timestamp": sim_time_str,
            "risk_level": "High",
            "recommended_action": rec_action,
            "confidence": 0.96,
            "status": "pending"
        }

    def create_civic_alert(self, civic_event: Dict[str, Any]) -> Dict[str, Any]:
        """Converts civic event (litter, dustbin, yellow line) to station master alert."""
        alert_type = civic_event.get("event_type", "littering")
        prefix_map = {
            "littering": "LIT",
            "dustbin full": "BIN",
            "yellow-line": "YEL",
            "queue congestion": "QUE"
        }
        prefix = prefix_map.get(alert_type, "CIV")
        t_str = civic_event.get("timestamp", "10:00").replace(":", "")
        loc_slug = civic_event.get("location", "P1").replace(" ", "")
        alert_id = f"{prefix}-{loc_slug}-{t_str}"

        db.log_alert(
            alert_id=alert_id,
            alert_type=alert_type,
            location=civic_event.get("location", "P1"),
            sim_timestamp=civic_event.get("timestamp", "10:00"),
            risk_level=civic_event.get("risk_level", "Medium"),
            recommended_action=civic_event.get("recommended_action", "Investigate and resolve"),
            confidence=0.93,
            db_path=self.db_path
        )
        db.log_event(
            event_type=f"civic alert: {alert_type}",
            location=civic_event.get("location", "P1"),
            details=civic_event.get("details", ""),
            sim_timestamp=civic_event.get("timestamp", "10:00"),
            db_path=self.db_path
        )
        return {
            "id": alert_id,
            "type": alert_type,
            "location": civic_event.get("location", "P1"),
            "timestamp": civic_event.get("timestamp", "10:00"),
            "risk_level": civic_event.get("risk_level", "Medium"),
            "recommended_action": civic_event.get("recommended_action", "Investigate"),
            "status": "pending"
        }

    def handle_operator_decision(
        self,
        alert_id: str,
        decision: str,  # 'approved', 'rejected', 'overridden'
        override_reason: Optional[str] = None,
        operator_id: str = "SM_DUTY"
    ) -> bool:
        """Executes human station master approval, rejection, or override."""
        success = db.update_alert_status(
            alert_id=alert_id,
            status=decision,
            operator_id=operator_id,
            override_reason=override_reason,
            completed=1 if decision == "approved" else 0,
            db_path=self.db_path
        )
        return success

    def mark_completed(self, alert_id: str) -> bool:
        return db.mark_alert_completed(alert_id, db_path=self.db_path)

_GLOBAL_ALERT_ENGINE = None

def get_alert_engine() -> AlertEngine:
    global _GLOBAL_ALERT_ENGINE
    if _GLOBAL_ALERT_ENGINE is None:
        _GLOBAL_ALERT_ENGINE = AlertEngine()
    return _GLOBAL_ALERT_ENGINE
