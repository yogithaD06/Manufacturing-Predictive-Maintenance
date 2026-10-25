
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser(description="Generate the manufacturing predictive-maintenance dataset")
ap.add_argument("--machines", type=int, default=120, help="number of machines (default 120)")
ap.add_argument("--days", type=int, default=120, help="days of telemetry (default 120)")
ap.add_argument("--out", default="data", help="output folder (default ./data)")
ap.add_argument("--csv", action="store_true", help="also write the large sensor_readings.csv")
ARGS = ap.parse_args()

RNG = np.random.default_rng(42)
OUT = Path(ARGS.out)
OUT.mkdir(exist_ok=True)

N_MACHINES = ARGS.machines
START = pd.Timestamp("2025-01-01")
DAYS = ARGS.days
FREQ_MIN = 10                       
STEPS = DAYS * 24 * 60 // FREQ_MIN  

PLANTS = {
    "PLT-CHN": ("Chennai", "Tamil Nadu"),
    "PLT-PUN": ("Pune", "Maharashtra"),
    "PLT-BLR": ("Bengaluru", "Karnataka"),
    "PLT-AHM": ("Ahmedabad", "Gujarat"),
}
TYPES = {
    "CNC_Lathe":          (55, 2.0, 6.0, 2400, 15, 1.0),
    "Hydraulic_Press":    (60, 3.0, 180., 0,   45, 1.2),
    "Air_Compressor":     (75, 4.0, 8.5, 1500, 37, 1.4),
    "Conveyor_Motor":     (45, 1.5, 0.0, 1450, 7.5, 0.8),
    "Injection_Molding":  (210, 2.5, 140., 120, 55, 1.1),
}
MANUFACTURERS = ["Bharat Machines", "Deccan Tools", "Sahyadri Engg", "Kaveri Hydraulics", "Narmada Motors"]
FAILURE_MODES = {
    "CNC_Lathe": ["Spindle_Bearing_Wear", "Tool_Breakage", "Coolant_Pump_Failure"],
    "Hydraulic_Press": ["Seal_Leak", "Hydraulic_Pump_Failure", "Valve_Stuck"],
    "Air_Compressor": ["Overheating", "Bearing_Failure", "Valve_Failure"],
    "Conveyor_Motor": ["Belt_Misalignment", "Motor_Winding_Failure", "Bearing_Failure"],
    "Injection_Molding": ["Heater_Band_Failure", "Screw_Wear", "Hydraulic_Leak"],
}


types = list(TYPES)
machines = pd.DataFrame({
    "machine_id": [f"M{i:04d}" for i in range(1, N_MACHINES + 1)],
    "plant_id": RNG.choice(list(PLANTS), N_MACHINES),
    "machine_type": RNG.choice(types, N_MACHINES),
    "manufacturer": RNG.choice(MANUFACTURERS, N_MACHINES),
    "install_year": RNG.integers(2008, 2024, N_MACHINES),
})
machines["city"] = machines.plant_id.map(lambda p: PLANTS[p][0])
machines["state"] = machines.plant_id.map(lambda p: PLANTS[p][1])
machines["rated_power_kw"] = machines.machine_type.map(lambda t: TYPES[t][4])
machines["age_years"] = 2025 - machines.install_year
machines["criticality"] = RNG.choice(["High", "Medium", "Low"], N_MACHINES, p=[.3, .5, .2])
machines.to_csv(OUT / "machines.csv", index=False)


ts = pd.date_range(START, periods=STEPS, freq=f"{FREQ_MIN}min")
hour = ts.hour.values
dow = ts.dayofweek.values
load_profile = np.where((hour >= 6) & (hour < 22), 0.85, 0.55) * np.where(dow == 6, 0.5, 1.0)

sensor_frames, failures, maint = [], [], []
fail_id = maint_id = 0
for m in machines.itertuples():
    bt, bv, bp, br, bw, fr = TYPES[m.machine_type]
    age_factor = 1 + m.age_years / 20
    health = np.empty(STEPS)
    h = RNG.uniform(0.75, 1.0)
    decay = RNG.uniform(6e-5, 1.4e-4) * fr * age_factor
    next_pm = RNG.integers(0, 30 * 144) 
    status = np.zeros(STEPS, dtype=np.int8)  
    down_until = -1
    for t in range(STEPS):
        if t < down_until:
            health[t] = h
            continue
        load = load_profile[t]
        h -= decay * (0.5 + load) * RNG.lognormal(0, 0.3)
        if RNG.random() < 2e-5 * fr:
            h -= RNG.uniform(0.05, 0.2)
        if t >= next_pm:
            dur = int(RNG.integers(12, 36)) 
            status[t:t + dur] = 3
            h = min(1.0, h + RNG.uniform(0.15, 0.35))
            maint_id += 1
            maint.append((f"MT{maint_id:06d}", m.machine_id, ts[t], ts[min(t + dur, STEPS - 1)],
                          "Preventive", RNG.choice(["Lubrication", "Inspection", "Filter_Change", "Calibration", "Belt_Tension"]),
                          round(dur * FREQ_MIN / 60, 2), int(RNG.integers(2000, 15000)),
                          f"TECH{RNG.integers(1, 40):03d}"))
            next_pm = t + int(RNG.normal(30 * 144, 3 * 144))
            down_until = t + dur
        elif h < 0.2:
            dur = int(RNG.integers(36, 216)) 
            status[t:t + dur] = 2
            mode = RNG.choice(FAILURE_MODES[m.machine_type])
            fail_id += 1
            failures.append((f"F{fail_id:05d}", m.machine_id, ts[t], mode,
                             RNG.choice(["Minor", "Major", "Critical"], p=[.4, .45, .15]),
                             round(dur * FREQ_MIN / 60, 2)))
            maint_id += 1
            maint.append((f"MT{maint_id:06d}", m.machine_id, ts[t], ts[min(t + dur, STEPS - 1)],
                          "Corrective", f"Repair_{mode}", round(dur * FREQ_MIN / 60, 2),
                          int(RNG.integers(15000, 250000)), f"TECH{RNG.integers(1, 40):03d}"))
            h = RNG.uniform(0.85, 1.0)
            down_until = t + dur
        health[t] = h

    running = status == 0
    idle = running & (RNG.random(STEPS) < (1 - load_profile) * 0.3)
    status[idle] = 1
    on = (status == 0).astype(float)
    deg = 1 - health 
    ambient = 28 + 5 * np.sin(2 * np.pi * (np.arange(STEPS) / 144 - 0.25)) + 4 * np.sin(2 * np.pi * np.arange(STEPS) / (144 * 365) * 1)
    temp = ambient + on * (bt - 30) * (0.7 + 0.3 * load_profile) * (1 + 0.6 * deg ** 2) + RNG.normal(0, 1.2, STEPS)
    vib = on * bv * (1 + 3.0 * deg ** 3) * load_profile + RNG.gamma(2, 0.08, STEPS)
    pres = on * bp * (1 - 0.25 * deg ** 2) + RNG.normal(0, bp * 0.02 + 0.05, STEPS)
    rpm = on * br * (1 - 0.08 * deg ** 2) + RNG.normal(0, br * 0.005 + 0.1, STEPS)
    power = (on * bw * load_profile * (1 + 0.35 * deg ** 2) + (status == 1) * bw * 0.1
             + RNG.normal(0, bw * 0.02, STEPS))
    acoustic = 55 + on * (20 + 15 * deg ** 2) + RNG.normal(0, 2, STEPS)
    oil_q = np.clip(100 * health + RNG.normal(0, 3, STEPS), 0, 100)

    df = pd.DataFrame({
        "timestamp": ts, "machine_id": m.machine_id,
        "temperature_c": temp.round(2), "vibration_mm_s": np.clip(vib, 0, None).round(3),
        "pressure_bar": np.clip(pres, 0, None).round(2), "rpm": np.clip(rpm, 0, None).round(1),
        "power_kw": np.clip(power, 0, None).round(2), "acoustic_db": acoustic.round(1),
        "oil_quality_index": oil_q.round(1),
        "machine_status": pd.Categorical.from_codes(status, ["Running", "Idle", "Failed", "Maintenance"]),
        "health_index": health.round(4),
    })
    for col in ["temperature_c", "vibration_mm_s", "pressure_bar", "power_kw"]:
        miss = RNG.random(STEPS) < 0.005
        df.loc[miss, col] = np.nan
    spikes = RNG.random(STEPS) < 0.0005
    df.loc[spikes, "vibration_mm_s"] *= RNG.uniform(3, 6, spikes.sum())
    sensor_frames.append(df)
    print(f"{m.machine_id} done, failures so far {fail_id}", end="\r")

sensors = pd.concat(sensor_frames, ignore_index=True)
fail_df = pd.DataFrame(failures, columns=["failure_id", "machine_id", "failure_time", "failure_mode", "severity", "downtime_hours"])
maint_df = pd.DataFrame(maint, columns=["maintenance_id", "machine_id", "start_time", "end_time", "maintenance_type",
                                        "action", "duration_hours", "cost_inr", "technician_id"])

sensors["failure_within_24h"] = 0
sensors["failure_within_7d"] = 0
sensors["rul_hours"] = np.nan 
for mid, g in fail_df.groupby("machine_id"):
    mask = (sensors.machine_id == mid).values
    t_arr = sensors.loc[mask, "timestamp"].values
    ft = np.sort(g.failure_time.values)
    pos = np.searchsorted(ft, t_arr, side="left")
    has_next = pos < len(ft)
    nxt = np.where(has_next, ft[np.minimum(pos, len(ft) - 1)], np.datetime64("NaT", "ns"))
    delta_h = (nxt - t_arr) / np.timedelta64(1, "h")
    rows = np.where(mask)[0]
    sensors.loc[rows, "rul_hours"] = np.round(delta_h, 2)
    sensors.loc[rows, "failure_within_24h"] = ((delta_h >= 0) & (delta_h <= 24)).astype(int)
    sensors.loc[rows, "failure_within_7d"] = ((delta_h >= 0) & (delta_h <= 168)).astype(int)


shift_rows = []
for m in machines.itertuples():
    s = sensors[sensors.machine_id == m.machine_id]
    s = s.assign(date=s.timestamp.dt.date,
                 shift=pd.cut(s.timestamp.dt.hour, [-1, 5, 13, 21, 24], labels=["C", "A", "B", "C2"]).astype(str).str[0])
    g = s.groupby(["date", "shift"])
    agg = g.agg(run=("machine_status", lambda x: (x == "Running").sum()),
                health=("health_index", "mean")).reset_index()
    agg["machine_id"] = m.machine_id
    agg["planned_minutes"] = 480
    agg["run_minutes"] = np.minimum(agg.run * FREQ_MIN, 480)
    rate = {"CNC_Lathe": 1.2, "Hydraulic_Press": 4, "Air_Compressor": 0, "Conveyor_Motor": 0, "Injection_Molding": 2.5}[m.machine_type]
    agg["units_produced"] = (agg.run_minutes * rate * RNG.uniform(0.85, 1.0, len(agg))).astype(int)
    rej_rate = np.clip(0.01 + 0.08 * (1 - agg.health) ** 2 + RNG.normal(0, 0.004, len(agg)), 0, 1)
    agg["units_rejected"] = (agg.units_produced * rej_rate).astype(int)
    agg["operator_id"] = [f"OP{x:03d}" for x in RNG.integers(1, 200, len(agg))]
    shift_rows.append(agg.drop(columns=["run", "health"]))
shifts = pd.concat(shift_rows, ignore_index=True).rename(columns={"date": "shift_date", "shift": "shift_code"})
shifts["availability_pct"] = (100 * shifts.run_minutes / shifts.planned_minutes).round(2)


sensors.to_parquet(OUT / "sensor_readings.parquet", index=False)
if ARGS.csv:
    sensors.to_csv(OUT / "sensor_readings.csv", index=False)
fail_df.to_csv(OUT / "failures.csv", index=False)
maint_df.to_csv(OUT / "maintenance_logs.csv", index=False)
shifts.to_csv(OUT / "production_shifts.csv", index=False)
print(f"\nmachines={len(machines)} sensors={len(sensors):,} failures={len(fail_df)} maint={len(maint_df)} shifts={len(shifts):,}")
