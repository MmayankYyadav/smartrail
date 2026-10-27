"""
SmartRail - AI-Assisted Railway Station Decision-Support Dashboard
Streamlit Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time

# SmartRail modules
from sim import get_station_simulation, DEFAULT_TRAINS
from delay_model import get_delay_predictor
from allocator import PLATFORM_LENGTHS, DWELL_TIMES
from vision import get_vision_counter, ensure_sample_video_exists, SAMPLE_VIDEO_PATH
from civic import CivicDisciplineManager
from alerts import compute_crowd_risk, get_alert_engine
from announce import (
    generate_coach_guidance_board,
    build_announcement,
    synthesize_voice,
    TEMPLATES
)
import db

# Page Configuration
st.set_page_config(
    page_title="SmartRail - Station Decision Support",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Main theme styling */
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 5px solid #0056b3;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .headline-action-yes {
        background: #ffebe6;
        border: 2px solid #ff4d4f;
        border-radius: 8px;
        padding: 16px;
        color: #cf1322;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .headline-action-no {
        background: #f6ffed;
        border: 2px solid #52c41a;
        border-radius: 8px;
        padding: 16px;
        color: #389e0d;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .station-display-board {
        background-color: #0d1117;
        color: #ffb703;
        font-family: 'Courier New', Courier, monospace;
        padding: 16px;
        border-radius: 8px;
        border: 3px solid #30363d;
        box-shadow: inset 0 0 10px #000;
        margin: 10px 0;
    }
    .guidance-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
    }
    .status-badge-green {
        background-color: #d1fae5;
        color: #065f46;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-badge-amber {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-badge-red {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "sim" not in st.session_state:
    st.session_state.sim = get_station_simulation()
if "vision_counter" not in st.session_state:
    st.session_state.vision_counter = get_vision_counter()
if "alert_engine" not in st.session_state:
    st.session_state.alert_engine = get_alert_engine()
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "audio_format" not in st.session_state:
    st.session_state.audio_format = "audio/wav"
if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = ""

sim = st.session_state.sim
alert_engine = st.session_state.alert_engine

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/45/Indian_Railways_logo.svg/200px-Indian_Railways_logo.svg.png", width=70)
st.sidebar.title("SmartRail Console")
st.sidebar.caption("Railway Station Decision-Support System")

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Zone Capacities (Persons)")
col_cap_a, col_cap_b, col_cap_c = st.sidebar.columns(3)
with col_cap_a:
    cap_a = st.number_input("Zone A", min_value=20, max_value=200, value=sim.zone_capacities["A"], step=5)
with col_cap_b:
    cap_b = st.number_input("Zone B", min_value=20, max_value=200, value=sim.zone_capacities["B"], step=5)
with col_cap_c:
    cap_c = st.number_input("Zone C", min_value=20, max_value=200, value=sim.zone_capacities["C"], step=5)
sim.set_zone_capacities(cap_a, cap_b, cap_c)

st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Vision Zone Counter")
use_simulated_vision = st.sidebar.toggle("Use simulated counts", value=True, help="When ON, crowd counts follow timetable arrivals. When OFF, runs YOLOv8n person detection on crowd video.")

if not use_simulated_vision:
    st.sidebar.info("Camera Stream Mode: Ultralytics YOLOv8n (person class only). No video frames stored.")
    ensure_sample_video_exists()
    if st.sidebar.button("Run YOLO Frame Analysis"):
        with st.spinner("Analyzing crowd frame with YOLOv8n..."):
            counts, inflows = st.session_state.vision_counter.process_video_frame(SAMPLE_VIDEO_PATH)
            sim.crowd_counts["P1"] = counts
            sim.inflows["P1"] = inflows
            st.sidebar.success(f"P1 Counts: A={counts['A']}, B={counts['B']}, C={counts['C']}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ What-If Scenario Injections")
st.sidebar.caption("Test system decision support under stress:")

if st.sidebar.button("⚡ Inject 12m delay on Train 12951", use_container_width=True):
    sim.inject_train_delay("12951", 12.0)
    st.sidebar.success("Injected +12m delay on 12951 Mumbai Rajdhani!")
    st.rerun()

p2_is_blocked = "P2" in sim.blocked_platforms
block_label = "🔓 Unblock Platform 2" if p2_is_blocked else "🚧 Block Platform 2 (Maintenance)"
if st.sidebar.button(block_label, use_container_width=True):
    sim.toggle_platform_block("P2")
    st.rerun()

if st.sidebar.button("👥 Crowd surge in Zone C (P3)", use_container_width=True):
    sim.trigger_crowd_surge("P3", "C")
    sim.civic_mgr.trigger_litter_event("P3", "C")
    st.sidebar.warning("Injected crowd surge & litter event on P3 Zone C!")
    st.rerun()

if st.sidebar.button("🔄 Reset Station State", use_container_width=True):
    sim.reset()
    st.sidebar.info("Station simulation reset to 09:55 AM.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.75rem; color:#6b7280; line-height:1.4;">
<b>Safety & Ethics Boundary:</b><br>
• Advisory only: Station Master approves or rejects.<br>
• Never directly signals trains or sets points.<br>
• Never issues automated fines or uses face recognition.<br>
• Vision counts headcounts only; frames never saved.
</div>
""", unsafe_allow_html=True)


# --- TOP MAIN HEADER & CLOCK ---
header_col1, header_col2, header_col3 = st.columns([5, 3, 4])
with header_col1:
    st.title("🚆 SmartRail")
    st.caption("AI-Assisted Railway Station Decision-Support Dashboard | Fixed Seed 42 | Offline Mode")

with header_col2:
    st.markdown(f"""
    <div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:10px 14px; text-align:center;">
        <span style="font-size:0.8rem; color:#4338ca; font-weight:600;">SIMULATION CLOCK</span><br>
        <span style="font-size:1.6rem; font-weight:700; color:#1e1b4b;">🕒 {sim.current_time_str}</span>
        <div style="font-size:0.75rem; color:#6366f1;">Step: +{sim.step_count} mins</div>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    step_col1, step_col2 = st.columns(2)
    with step_col1:
        if st.button("⏩ Step +1 min", use_container_width=True, type="primary"):
            sim.step(1)
            st.rerun()
    with step_col2:
        auto_toggle = st.toggle("Auto-run", value=st.session_state.auto_run)
        st.session_state.auto_run = auto_toggle

    speed_sec = st.slider("Step speed (seconds)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

# Banner alerts if platform blocked or surge active
if sim.blocked_platforms:
    st.warning(f"⚠️ OPERATIONAL NOTICE: Platform(s) {', '.join(sim.blocked_platforms)} are BLOCKED for maintenance. Trains rerouted to alternative platforms.")

# Main Navigation Tabs
tab_ops, tab_safety, tab_civic, tab_metrics = st.tabs([
    "🚉 Operations & Timetable",
    "🛡️ Passenger Safety & Crowd Risk",
    "🧹 Civic Discipline & Sanitation",
    "📊 Logs & Model Metrics"
])


# ==============================================================================
# TAB 1: OPERATIONS & TIMETABLE
# ==============================================================================
with tab_ops:
    st.subheader("📋 Live Station Train Board (Timetable vs Decision-Support)")
    st.caption("XGBoost delay predictions combined with rule-based platform allocation engine.")

    # Format trains table
    train_rows = []
    for t in sim.trains:
        is_changed = t.get("is_platform_changed", False)
        plat_display = f"⚠️ {t.get('recommended_platform')} (was {t.get('usual_platform')})" if is_changed else f"✅ {t.get('recommended_platform')}"
        
        train_rows.append({
            "Train No.": t["number"],
            "Train Name": t["name"],
            "Type": t["type"],
            "Length": f"{t['length']} coaches",
            "Priority": f"P{t['priority']}",
            "Sched Arr": t["scheduled_arrival"],
            "Pred Delay": f"+{t.get('predicted_delay', 0.0):.1f} min",
            "Pred Arr": t.get("predicted_arrival", t["scheduled_arrival"]),
            "Assigned Plat": plat_display,
            "Allocation Reason": t.get("reason", "Scheduled platform allocated")
        })

    df_trains = pd.DataFrame(train_rows)
    st.dataframe(
        df_trains,
        column_config={
            "Train No.": st.column_config.TextColumn(width="small"),
            "Train Name": st.column_config.TextColumn(width="medium"),
            "Allocation Reason": st.column_config.TextColumn(width="large")
        },
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("📊 Platform Timeline (Occupancy & Conflict Detection)")
    st.caption("Visual Gantt-style platform occupancy bars. Dwell + 5 min cleaning buffer shown.")

    # Build Plotly Gantt Chart for Platforms
    timeline_data = []
    base_date = datetime(2026, 10, 27)

    for t in sim.trains:
        p_alloc = t.get("recommended_platform", "P1")
        arr_t = datetime.strptime(t.get("predicted_arrival", "10:00"), "%H:%M")
        dwell = DWELL_TIMES.get(t["type"], 8) + 5
        dep_t = arr_t + timedelta(minutes=dwell)

        start_dt = datetime(2026, 10, 27, arr_t.hour, arr_t.minute)
        end_dt = datetime(2026, 10, 27, dep_t.hour, dep_t.minute)

        timeline_data.append({
            "Platform": p_alloc,
            "Train": f"{t['number']} {t['name']}",
            "Type": t["type"],
            "Start": start_dt,
            "End": end_dt,
            "Priority": f"Priority {t['priority']}"
        })

    df_timeline = pd.DataFrame(timeline_data)
    if not df_timeline.empty:
        fig_timeline = px.timeline(
            df_timeline,
            x_start="Start",
            x_end="End",
            y="Platform",
            color="Type",
            hover_name="Train",
            category_orders={"Platform": ["P1", "P2", "P3"]},
            color_discrete_map={"Rajdhani": "#ef4444", "Express": "#3b82f6", "Local": "#10b981"},
            title="Current Platform Occupancy Schedule"
        )
        fig_timeline.update_yaxes(autorange="reversed")
        fig_timeline.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")
    st.subheader("📢 Announcement & Operational Action Approval")
    st.caption("Review system-drafted passenger announcements before broadcast. Station Master approves or rejects.")

    # Find trains with delay or platform change
    notable_trains = [t for t in sim.trains if t.get("is_platform_changed") or t.get("predicted_delay", 0) >= 5.0]
    selected_train = st.selectbox(
        "Select train for announcement draft:",
        options=sim.trains,
        format_func=lambda t: f"{t['number']} - {t['name']} ({t.get('recommended_platform')}, +{t.get('predicted_delay', 0.0):.1f}m)"
    )

    if selected_train:
        ann_col1, ann_col2 = st.columns([7, 5])
        
        # Decide template type
        if selected_train.get("is_platform_changed"):
            ann_type = "platform_change"
            params = {
                "train_num": selected_train["number"],
                "train_name": selected_train["name"],
                "old_platform": selected_train["usual_platform"],
                "new_platform": selected_train["recommended_platform"]
            }
        else:
            ann_type = "delay"
            params = {
                "train_num": selected_train["number"],
                "train_name": selected_train["name"],
                "delay_mins": int(selected_train.get("predicted_delay", 0)),
                "arrival_time": selected_train.get("predicted_arrival", "10:00")
            }

        text_en, text_hi = build_announcement(ann_type, params)

        with ann_col1:
            st.markdown(f"**Draft Announcement ({ann_type.replace('_', ' ').title()}):**")
            st.info(f"🇬🇧 **English:** {text_en}\n\n🇮🇳 **Hindi:** {text_hi}")

        with ann_col2:
            st.markdown("**Station Master Action:**")
            app_btn_col, rej_btn_col = st.columns(2)
            with app_btn_col:
                if st.button("✅ Approve Broadcast", key=f"app_ann_{selected_train['number']}", use_container_width=True):
                    db.log_event("announcement", selected_train.get("recommended_platform", "P1"), f"Approved announcement for {selected_train['number']}", sim.current_time_str)
                    st.success("Announcement broadcast approved and logged!")
            with rej_btn_col:
                if st.button("❌ Reject", key=f"rej_ann_{selected_train['number']}", use_container_width=True):
                    db.log_event("announcement", selected_train.get("recommended_platform", "P1"), f"Rejected announcement for {selected_train['number']}", sim.current_time_str)
                    st.warning("Announcement dismissed.")

            if st.button("🔊 Play Voice / Chime", key="play_voice_btn", use_container_width=True):
                with st.spinner("Synthesizing audio..."):
                    audio_data, audio_format = synthesize_voice(text_en, lang="en")
                    st.session_state.audio_bytes = audio_data
                    st.session_state.audio_format = audio_format
                    st.session_state.last_spoken_text = text_en

            if st.session_state.audio_bytes:
                st.audio(st.session_state.audio_bytes, format=st.session_state.audio_format)
                st.caption(f"Audio ready: '{st.session_state.last_spoken_text[:60]}...'")


# ==============================================================================
# TAB 2: PASSENGER SAFETY & CROWD RISK
# ==============================================================================
with tab_safety:
    st.subheader("🛡️ Platform Crowd Risk Matrix & Human Action Recommendations")
    st.caption("Formulas: predicted_ratio = (current_count + inflow_per_min * minutes_to_arrival) / zone_capacity")

    # 3x3 Crowd Heatmap (Platforms x Zones)
    heatmap_z = []
    hover_texts = []
    platforms = ["P1", "P2", "P3"]
    zones = ["A", "B", "C"]

    for p in platforms:
        row_z = []
        row_hover = []
        for z in zones:
            cnt = sim.crowd_counts[p][z]
            inflow = sim.inflows[p][z]
            cap = sim.zone_capacities[z]
            metric = compute_crowd_risk(z, cnt, inflow, minutes_to_arrival=12, zone_capacity=cap)
            
            # Numeric score for heatmap: 0 to 100
            score = metric["risk_score"]
            row_z.append(score)
            row_hover.append(f"{p} Zone {z}<br>Count: {cnt}/{cap}<br>Inflow: +{inflow}/m<br>Ratio: {metric['predicted_ratio']:.2f}<br>Risk: {metric['risk_level']}")
        heatmap_z.append(row_z)

    heat_col1, heat_col2 = st.columns([6, 6])

    with heat_col1:
        st.markdown("**3x3 Platform-Zone Crowd Density Heatmap**")
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_z,
            x=["Zone A (Front)", "Zone B (Middle)", "Zone C (Rear)"],
            y=["Platform 1", "Platform 2", "Platform 3"],
            hoverongaps=False,
            text=[[f"{val}%" for val in r] for r in heatmap_z],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale=[
                [0.0, "#10b981"],  # Green
                [0.5, "#f59e0b"],  # Amber
                [0.8, "#ef4444"],  # Red
                [1.0, "#991b1b"]   # Dark Red
            ],
            zmin=0,
            zmax=120
        ))
        fig_heat.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    with heat_col2:
        st.markdown("**Selected Platform Detailed Metrics:**")
        selected_p = st.selectbox("Inspect Platform:", ["P1", "P2", "P3"], index=2 if sim.surge_platform == "P3" else 0)
        
        c_cols = st.columns(3)
        zone_metrics = {}
        for idx, z in enumerate(zones):
            cnt = sim.crowd_counts[selected_p][z]
            inflow = sim.inflows[selected_p][z]
            cap = sim.zone_capacities[z]
            metric = compute_crowd_risk(z, cnt, inflow, minutes_to_arrival=12, zone_capacity=cap)
            zone_metrics[z] = metric

            with c_cols[idx]:
                risk_color = "#10b981" if metric["risk_level"] == "Low" else ("#f59e0b" if metric["risk_level"] == "Medium" else "#ef4444")
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {risk_color};">
                    <span style="font-weight:700;">Zone {z} ({'General' if z in ('A','C') else 'AC/Sleeper'})</span><br>
                    <span style="font-size:1.3rem; font-weight:800;">{cnt}</span> <span style="font-size:0.8rem; color:#6b7280;">/ {cap} pax</span><br>
                    <div style="font-size:0.75rem; margin-top:4px;">
                        Inflow: <b>+{inflow}/min</b><br>
                        Pred Ratio: <b>{metric['predicted_ratio']:.2f}</b><br>
                        Clearance: <b>~{metric['clearance_time_mins']} mins</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # HEADLINE YES/NO ACTIONS
    st.markdown("---")
    st.subheader("🎯 Station Master Crowd Action Directives")
    st.caption("Mandatory human decision point. System outputs explicit YES/NO directive with confidence score.")

    action_cols = st.columns(3)
    for idx, p in enumerate(platforms):
        with action_cols[idx]:
            # Check maximum risk across zones for this platform
            max_risk_zone = "C"
            max_score = -1
            worst_metric = None
            for z in zones:
                m = compute_crowd_risk(z, sim.crowd_counts[p][z], sim.inflows[p][z], minutes_to_arrival=12, zone_capacity=sim.zone_capacities[z])
                if m["risk_score"] > max_score:
                    max_score = m["risk_score"]
                    max_risk_zone = z
                    worst_metric = m

            is_yes = worst_metric["risk_level"] == "High"
            action_class = "headline-action-yes" if is_yes else "headline-action-no"
            badge_text = "YES - IMMEDIATE DISPATCH" if is_yes else "NO - NORMAL MONITORING"

            st.markdown(f"""
            <div class="{action_class}">
                <div style="font-size:0.85rem; text-transform:uppercase;">Platform {p} Directive</div>
                <div style="font-size:1.2rem; font-weight:800; margin:6px 0;">
                    Send staff to {p} Zone {max_risk_zone}? <br><span style="text-decoration:underline;">{worst_metric['staff_action_yes_no']}</span>
                </div>
                <div style="font-size:0.8rem; font-weight:normal;">
                    Peak Zone: {max_risk_zone} | Risk Score: {worst_metric['risk_score']}/100<br>
                    Headcount: {worst_metric['current_count']} pax (Capacity: {worst_metric['zone_capacity']})
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_yes:
                if st.button(f"👮 Approve Staff Dispatch to {p}", key=f"disp_staff_{p}", use_container_width=True, type="primary"):
                    alert_engine.handle_operator_decision(
                        alert_id=f"CRW-{p}{max_risk_zone}-{sim.current_time_str.replace(':', '')}",
                        decision="approved",
                        operator_id="SM_ON_DUTY"
                    )
                    db.log_event("staff dispatch", f"{p} Zone {max_risk_zone}", f"Station Master approved crowd marshal dispatch to {p} Zone {max_risk_zone}", sim.current_time_str)
                    st.success(f"Dispatched marshals to Platform {p} Zone {max_risk_zone}!")

    # Coach-zone guidance board panel
    st.markdown("---")
    st.subheader("📺 Platform Guidance Display Board (Dynamic Passenger Divert)")
    st.caption("Picks the less crowded general zone (A or C) to balance platform flow. Instructs passengers to queue at the marked line.")

    disp_col1, disp_col2 = st.columns([8, 4])
    with disp_col1:
        # Guidance board data for active platform
        c_p = selected_p
        guidance = generate_coach_guidance_board(
            train_num="12951",
            train_name="Mumbai Rajdhani",
            platform=c_p,
            zone_a_risk=zone_metrics["A"]["risk_level"],
            zone_c_risk=zone_metrics["C"]["risk_level"],
            zone_a_count=sim.crowd_counts[c_p]["A"],
            zone_c_count=sim.crowd_counts[c_p]["C"]
        )

        st.markdown(f"""
        <div class="station-display-board">
            <div style="font-size:0.85rem; color:#8b949e; letter-spacing:1px;">INDIAN RAILWAYS PASSENGER INFORMATION SYSTEM - {c_p}</div>
            <div style="font-size:1.3rem; font-weight:700; color:#58a6ff; margin-top:6px;">{guidance['display_text_en']}</div>
            <div style="font-size:1.1rem; color:#e3b341; margin-top:4px;">{guidance['display_text_hi']}</div>
            <div style="font-size:0.9rem; color:#7ee787; margin-top:8px;">⚠️ {guidance['marked_line_instruction']}</div>
        </div>
        """, unsafe_allow_html=True)

    with disp_col2:
        st.markdown("**Guidance Broadcast:**")
        if st.button("📢 Push Guidance to Platform Display", key="push_guidance_btn", use_container_width=True):
            db.log_event("guidance display", c_p, f"Pushed coach guidance board to {c_p}: {guidance['target_zone']} recommended", sim.current_time_str)
            st.success("Guidance sent to overhead platform LED displays!")

    # Before / After Chart: Guidance Impact
    st.markdown("---")
    st.subheader("📈 Before / After Guidance Impact Analysis")
    st.caption("Simulates identical arrival scenario twice: with vs without guidance messages.")

    guidance_share = st.slider(
        "Assumed share of passengers following guidance (%)",
        min_value=5,
        max_value=60,
        value=25,
        step=5,
        help="Proportion of general-ticket passengers that shift from congested Zone C to less crowded Zone A when guidance is shown."
    )

    exp_data = sim.run_guidance_comparison_experiment(
        share_following_guidance=guidance_share / 100.0,
        total_minutes=35
    )

    fig_before_after = go.Figure()
    fig_before_after.add_trace(go.Scatter(
        x=exp_data["times"],
        y=exp_data["no_guidance_zone_c"],
        mode="lines+markers",
        name="Without Guidance (Zone C Unregulated)",
        line=dict(color="#ef4444", width=3, dash="dash")
    ))
    fig_before_after.add_trace(go.Scatter(
        x=exp_data["times"],
        y=exp_data["with_guidance_zone_c"],
        mode="lines+markers",
        name="With SmartRail Guidance (Zone C Regulated)",
        line=dict(color="#10b981", width=3)
    ))
    # Safe capacity threshold line
    fig_before_after.add_hline(y=1.0, line_dash="dot", line_color="#b91c1c", annotation_text="Platform Zone Capacity (1.0 Ratio)")
    fig_before_after.add_hline(y=0.8, line_dash="dot", line_color="#f59e0b", annotation_text="High Risk Threshold (0.8 Ratio)")

    fig_before_after.update_layout(
        title=f"General Zone C Peak Crowd Ratio Over Time ({exp_data['peak_reduction_pct']}% Peak Reduction)",
        xaxis_title="Simulation Time Progression",
        yaxis_title="Crowd / Zone Capacity Ratio",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_before_after, use_container_width=True)
    st.markdown('<span class="guidance-badge">simulated, assumption-based</span>', unsafe_allow_html=True)


# ==============================================================================
# TAB 3: CIVIC DISCIPLINE & SANITATION
# ==============================================================================
with tab_civic:
    st.subheader("🧹 Civic Discipline, Sanitation & Platform Cleanliness")
    st.caption("Automated tracking of dustbin levels, littering, yellow-line violations, and queue congestion.")

    # Dustbin fill progress bars
    st.markdown("**Platform Dustbin Fill Levels (Alert at >= 85%):**")
    bin_cols = st.columns(3)
    for idx, p in enumerate(platforms):
        fill = sim.civic_mgr.dustbin_fill[p]
        with bin_cols[idx]:
            bar_color = "red" if fill >= 85 else ("orange" if fill >= 65 else "green")
            st.metric(f"Dustbin {p}", f"{fill:.1f}%", delta=">= 85% High Alert" if fill >= 85 else "Normal")
            st.progress(fill / 100.0)

    st.markdown("---")
    # Zone Discipline Scores & Cleaning Priority
    disc_col, clean_col = st.columns([6, 6])

    with disc_col:
        st.markdown("**Zone Discipline Scores (0 - 100)**")
        st.caption("100 base score. Deductions: -8 per litter, -15 per yellow-line crossing, -10 per queue surge, -5 if dustbin >= 85%.")

        disc_rows = []
        for p in platforms:
            for z in zones:
                score, color_cat = sim.civic_mgr.get_discipline_score(p, z)
                badge_html = f"<span class='status-badge-{color_cat}'>{score} / 100</span>"
                disc_rows.append({
                    "Platform": p,
                    "Zone": z,
                    "Type": "General" if z in ("A", "C") else "AC/Sleeper",
                    "Score": score,
                    "Category": color_cat.upper(),
                    "Litter Events": sim.civic_mgr.littering_counts.get(f"{p}-{z}", 0),
                    "Yellow-Line": sim.civic_mgr.yellow_line_counts.get(f"{p}-{z}", 0),
                    "Queue Surges": sim.civic_mgr.queue_congestion_counts.get(f"{p}-{z}", 0)
                })

        df_disc = pd.DataFrame(disc_rows)
        st.dataframe(df_disc, use_container_width=True, hide_index=True)

    with clean_col:
        st.markdown("**Cleaning Priority Ranking (Urgent Cleaning Tasks)**")
        st.caption("Sorted by dustbin fill and litter count. Approve to dispatch sanitation team.")

        priority_list = sim.civic_mgr.get_cleaning_priority_list()
        for item in priority_list[:4]: # top 4 urgent
            is_urgent = item["status"] == "Urgent Clean Needed"
            border_c = "#ef4444" if is_urgent else "#e5e7eb"
            
            c_task_col1, c_task_col2 = st.columns([8, 4])
            with c_task_col1:
                st.markdown(f"""
                <div style="border-left:4px solid {border_c}; background:#f9fafb; padding:8px 12px; margin-bottom:8px; border-radius:4px;">
                    <b>{item['location']}</b> | Status: <b>{item['status']}</b><br>
                    <span style="font-size:0.8rem; color:#4b5563;">
                        Dustbin: {item['dustbin_fill_pct']:.1f}% | Litter count: {item['litter_count']} | Urgency: {item['urgency_score']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with c_task_col2:
                if st.button("🧹 Dispatch Crew", key=f"clean_{item['platform']}_{item['zone']}", use_container_width=True):
                    sim.civic_mgr.execute_cleaning_task(item["platform"], item["zone"])
                    db.log_event("sanitation", item["location"], "Station Master dispatched cleaning sweep crew", sim.current_time_str)
                    st.success(f"Cleaning crew dispatched to {item['location']}!")
                    st.rerun()

    st.markdown("---")
    st.subheader("⚠️ Yellow-Line Crossing & Safety Incidents")
    yellow_events = [e for e in sim.civic_mgr.recent_events if e["event_type"] == "yellow-line"]
    if yellow_events:
        for y_evt in yellow_events[-3:]:
            st.error(f"🚨 **{y_evt['timestamp']} - {y_evt['location']}:** {y_evt['details']}. Recommendation: *{y_evt['recommended_action']}*")
    else:
        st.success("No active yellow-line safety infractions at this minute.")


# ==============================================================================
# TAB 4: LOGS & MODEL METRICS
# ==============================================================================
with tab_metrics:
    st.subheader("📊 Decision Support Metrics & Model Validation")
    
    # Delay model metrics card
    predictor = get_delay_predictor()
    m_info = predictor.metrics

    st.markdown("**Delay Predictor Model Performance vs Naive Baseline:**")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Model Architecture", m_info.get("model_type", "XGBoost"))
    with m_col2:
        st.metric("Model MAE", f"{m_info.get('model_mae', 0.0):.2f} mins", delta=f"-{m_info.get('mae_improvement_pct', 0.0):.1f}% vs Baseline")
    with m_col3:
        st.metric("Model RMSE", f"{m_info.get('model_rmse', 0.0):.2f} mins", delta=f"-{m_info.get('rmse_improvement_pct', 0.0):.1f}% vs Baseline")
    with m_col4:
        st.metric("Baseline MAE (Mean)", f"{m_info.get('baseline_mae', 0.0):.2f} mins")

    st.markdown("---")
    # Operator Approval Performance
    summary = db.get_metrics_summary()
    st.markdown("**Station Master Approval & Override Metrics:**")
    op_col1, op_col2, op_col3, op_col4, op_col5 = st.columns(5)
    with op_col1:
        st.metric("Total Alerts", summary["total_alerts"])
    with op_col2:
        st.metric("Approved", summary["approved"])
    with op_col3:
        st.metric("Rejected", summary["rejected"])
    with op_col4:
        st.metric("Overridden", summary["overridden"])
    with op_col5:
        st.metric("Override Rate", f"{summary['override_rate_pct']:.1f}%")

    st.markdown("---")
    st.subheader("📋 Active & Historical Alert Management")
    st.caption("Human Station Master reviews, approves, rejects, or overrides recommended actions.")

    all_alerts = db.get_alerts(limit=25)
    if all_alerts:
        for alt in all_alerts:
            alt_id = alt["id"]
            status = alt["status"]
            
            with st.expander(f"Alert [{alt['id']}] - {alt['type'].upper()} at {alt['location']} ({status.upper()})", expanded=(status == "pending")):
                st.write(f"**Timestamp:** {alt['timestamp']} | **Risk:** {alt['risk_level']} | **Confidence:** {alt['confidence']:.2f}")
                st.write(f"**Recommended Action:** {alt['recommended_action']}")
                
                if alt["override_reason"]:
                    st.info(f"**Override Reason:** {alt['override_reason']}")

                if status == "pending":
                    act_col1, act_col2, act_col3 = st.columns([3, 3, 6])
                    with act_col1:
                        if st.button("✅ Approve", key=f"app_{alt_id}", use_container_width=True):
                            db.update_alert_status(alt_id, "approved")
                            st.success(f"Alert {alt_id} approved!")
                            st.rerun()
                    with act_col2:
                        if st.button("❌ Reject", key=f"rej_{alt_id}", use_container_width=True):
                            db.update_alert_status(alt_id, "rejected")
                            st.warning(f"Alert {alt_id} rejected.")
                            st.rerun()
                    with act_col3:
                        with st.popover("⚙️ Override with Custom Reason"):
                            override_text = st.text_input("Enter operator override justification:", key=f"txt_{alt_id}")
                            if st.button("Submit Override", key=f"btn_sub_{alt_id}"):
                                if override_text.strip():
                                    db.update_alert_status(alt_id, "overridden", override_reason=override_text.strip())
                                    st.info(f"Alert {alt_id} overridden.")
                                    st.rerun()
                                else:
                                    st.error("Please provide a reason.")
                else:
                    st.caption(f"Status: {status.upper()} | Completed: {'Yes' if alt['completed'] else 'No'}")
    else:
        st.info("No alerts logged yet. System running smoothly.")

    st.markdown("---")
    st.subheader("📜 System Audit Event Log")
    events = db.get_events(limit=30)
    if events:
        df_events = pd.DataFrame(events)[["timestamp", "event_type", "location", "details"]]
        st.dataframe(df_events, use_container_width=True, hide_index=True)
    else:
        st.caption("No event logs recorded.")


# Handle auto-run timer if enabled
if st.session_state.auto_run:
    time.sleep(speed_sec)
    sim.step(1)
    st.rerun()
