# 🚆 SmartRail: AI-Assisted Railway Station Decision-Support Dashboard

SmartRail is an intelligent, offline-first decision-support prototype engineered for Indian Railway station masters. It combines machine learning delay prediction, rule-based platform allocation, computer vision crowd monitoring, civic discipline tracking, and bilingual passenger guidance into a single unified operations dashboard.

---

## 🏛️ Core Principles & Ethical Boundaries
- **Advisory Decision Support Only:** SmartRail **never** directly signals trains, never switches track points, and never issues automated fines. A human station master retains full control to approve, reject, or override any recommendation.
- **Privacy by Design:** Ultralytics YOLOv8n performs headcount tallies only. **Zero facial recognition** is used, and **no video frames or biometric images are ever saved to disk**.
- **Deterministic Safety Communications:** All passenger safety alerts and announcements follow fixed, vetted bilingual templates (English & Hindi) without uncontrolled generative LLM hallucinations.
- **100% Offline Capability:** Operates completely offline with synthetic seed-based data (Seed: 42) and local model inference.

---

## 📂 Project Structure

```text
smartrail/
├── app.py              # Main Streamlit dashboard with 4 operational tabs & sidebar
├── sim.py              # Station simulation engine, timetable state & scenario injector
├── delay_model.py      # XGBoost / GradientBoosting delay regressor vs naive baseline
├── allocator.py        # Rule-based platform conflict resolution & scoring engine
├── vision.py           # YOLOv8n person counter (vertical thirds: Zone A/B/C)
├── civic.py            # Civic discipline manager (littering, dustbins, yellow-lines)
├── alerts.py           # Crowd risk engine, headline YES/NO directives & alert lifecycle
├── announce.py         # Bilingual announcement templates & voice/chime audio engine
├── db.py               # SQLite storage (smartrail.db) for alerts, events & audit logs
├── data/               # Sample crowd video storage
├── requirements.txt    # Python dependencies
└── README.md           # Documentation and 2-minute demo walkthrough script
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10 to 3.14 on Windows, macOS, or Linux.
- No GPU required (CPU-optimized inference).

### 2. Install Dependencies
```bash
cd smartrail
pip install -r requirements.txt
```

### 3. Launch SmartRail
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⏱️ 2-Minute Guided Demo Script

Follow this step-by-step sequence to demonstrate the complete decision-support workflow:

### Step 1: Normal Day Operations (30 seconds)
1. Open the dashboard. Observe the **Simulation Clock** at `09:55 AM`.
2. In the **Operations & Timetable** tab:
   - Notice the **Live Train Board**: 8 trains with scheduled vs predicted arrival times, coach lengths, priorities, and assigned platforms.
   - Inspect the **Platform Timeline (Gantt Chart)** showing conflict-free platform occupancy bars with dwell times and 5-minute cleaning buffers.
3. Click **"Step +1 min"** a few times to advance the clock to `10:00 AM`. Notice realistic crowd build-up as trains approach.

### Step 2: Inject Train Delay & Observe Platform Conflict Resolution (30 seconds)
1. In the sidebar under **"What-If Scenario Injections"**, click:
   👉 **`⚡ Inject 12m delay on Train 12951`**
2. Notice immediate system reaction:
   - Train 12951 (Mumbai Rajdhani) arrival shifts from `10:00` to `10:14`.
   - Because 12951 now occupies Platform 1 until `10:29`, the allocator detects a conflict with Train 12301 (Howrah Rajdhani, arriving `10:25`).
   - The Allocator automatically reassigns Train 12301 to **Platform 3**, providing a plain-language reason:
     *`"P3 chosen: Scheduled P1 has train occupancy conflict; length fits (24 coaches), crowd risk Low."`*
3. Scroll down to **"Announcement & Operational Action Approval"**:
   - Inspect the generated bilingual platform change draft (English + Hindi).
   - Click **"🔊 Play Voice / Chime"** to hear the Indian Railways announcement chime.
   - Click **"✅ Approve Broadcast"** to log station master clearance.

### Step 3: Crowd Surge & Headline YES/NO Action Directive (25 seconds)
1. In the sidebar, click:
   👉 **`👥 Crowd surge in Zone C (P3)`**
2. Switch to the **"Passenger Safety & Crowd Risk"** tab:
   - Notice the **3x3 Crowd Density Heatmap** highlighting Platform 3 Zone C in red.
   - Look at the **Headline Directive Card**:
     ```
     Platform P3 Directive:
     Send staff to P3 Zone C? YES
     Peak Zone: C | Risk Score: 100/100
     ```
   - Click **"👮 Approve Staff Dispatch to P3"** to mobilize station marshals.
3. Observe the **Platform Guidance Display Board**:
   - It automatically identifies that Zone C is overcrowded and commands:
     *`"General coaches: Zone A is less crowded, please queue there at the marked line."`*
4. Scroll to the **Before / After Guidance Impact Analysis** chart:
   - Adjust the **"Assumed share of passengers following guidance"** slider (e.g. 25%).
   - See how the guided scenario prevents severe crushing by lowering peak crowd ratio by ~16% to 25%.

### Step 4: Civic Discipline & Sanitation Sweep (20 seconds)
1. Switch to the **"Civic Discipline & Sanitation"** tab:
   - Check the **Platform Dustbin Fill Levels**: Platform 3 dustbin has surpassed the `>= 85%` threshold.
   - Inspect the **Discipline Scores**: Zone C has dropped due to littering and crowd surge deductions.
2. In the **Cleaning Priority Ranking**, locate the urgent task for Platform 3.
3. Click **"🧹 Dispatch Crew"**:
   - Notice dustbin level empties and litter counter resets.
   - The zone's Discipline Score immediately recovers towards green.

### Step 5: Metrics & Audit Trail (15 seconds)
1. Switch to the **"Logs & Model Metrics"** tab:
   - **Delay Predictor Validation**: Note that the trained XGBoost model achieves an MAE of ~3.3 mins vs the 10.7 min naive baseline (~69% improvement).
   - **Station Master Performance**: Observe approved, rejected, and overridden alert counters, along with the override rate.
   - Test an **Override**: Expand any pending alert, click "Override with Custom Reason", enter an operator note (e.g. *"VIP movement priority"*), and confirm.
   - Check the **System Audit Event Log** table querying live SQLite data from `smartrail.db`.

---

## 🛡️ Station Model Details
- **Platforms:**
  - P1: Length 24 coaches
  - P2: Length 22 coaches (cannot accommodate 24-coach trains)
  - P3: Length 24 coaches
- **Zones:**
  - Zone A: Front (General / Unreserved)
  - Zone B: Middle (AC / Sleeper)
  - Zone C: Rear (General / Unreserved)
- **Train Fleet:**
  - 8 realistic trains spanning Rajdhani (Priority 1), Express (Priority 2), and Local Shuttles (Priority 3).
