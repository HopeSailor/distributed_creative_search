"""Construct the career and calendar windows used in the historical diagnostic."""

import re
import numpy as np
import pandas as pd

TEMP_RE = re.compile(r"study2_movement_(\d{4}H[12])_(history|focal)")


def _consecutive(g: pd.DataFrame) -> bool:
    idx = g.sort_values("created_at")["creator_sequence_index"].to_numpy(float)
    return bool(len(idx) > 1 and np.all(np.diff(idx) == 1))


def build_windows(df):
    windows: list[dict] = []
    # Career windows: one ten-video realized sequence for each creator-stage pair.
    for q in (20, 40, 60, 80):
        col = f"study1_q{q}_role"
        selected = df[df[col].notna() & df[col].astype(str).ne("")].copy()
        for creator, g in selected.groupby("creator_handle"):
            g = g.sort_values("creator_sequence_index")
            valid = len(g) == 10 and _consecutive(g)
            for pos, (_, row) in enumerate(g.iterrows(), 1):
                windows.append(
                    {
                        "window_family": "career",
                        "window_id": f"career::{creator}::q{q}",
                        "creator_handle": creator,
                        "period": f"q{q}",
                        "position": pos,
                        "content_id": str(row.content_id),
                        "created_at": row.created_at,
                        "creator_sequence_index": row.creator_sequence_index,
                        "planned_role": row[col],
                        "consecutive_window": valid,
                    }
                )
    # Calendar-time windows: explicit 8-history + focal roles in the frozen sample.
    temp_rows: list[dict] = []
    for _, row in df.iterrows():
        for match in TEMP_RE.finditer(str(row.get("selection_reasons", ""))):
            temp_rows.append({"period": match.group(1), "role": match.group(2), **row.to_dict()})
    td = pd.DataFrame(temp_rows)
    if not td.empty:
        for (creator, period), g in td.groupby(["creator_handle", "period"]):
            g = g.sort_values("creator_sequence_index").drop_duplicates("content_id")
            valid = len(g) == 9 and _consecutive(g)
            for pos, (_, row) in enumerate(g.iterrows(), 1):
                windows.append(
                    {
                        "window_family": "temporal",
                        "window_id": f"temporal::{creator}::{period}",
                        "creator_handle": creator,
                        "period": period,
                        "position": pos,
                        "content_id": str(row.content_id),
                        "created_at": row.created_at,
                        "creator_sequence_index": row.creator_sequence_index,
                        "planned_role": row.role,
                        "consecutive_window": valid,
                    }
                )
    window_members = pd.DataFrame(windows)
    return window_members
