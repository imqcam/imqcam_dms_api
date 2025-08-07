from functools import lru_cache
from girder_client import GirderClient
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import os
from periodictable import Al, V, Cr, Mn, Fe, Co, Ni, Cu
import logging
import re
from collections import defaultdict
import json
from datetime import timedelta, datetime
import folium
from geopy.distance import geodesic
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative

def build_timeline(client, filter_by_location):
    
    # Replace this with your actual sample ID list
    all_samples = client.get("sample") 
    sample_ids = [sample["_id"] for sample in all_samples]

    # 1. Fetch and structure data
    all_events = []

    for sid in sample_ids:
        raw_data = client.get("sample/id", parameters={"id": sid})
        sample_name = raw_data.get("name")
        print(sample_name)
        for event in raw_data.get("events", []):
            all_events.append({
                "sample_id": sid,
                "sample_name": sample_name,
                "event_type": event.get("eventType"),
                "creator": event.get("creatorName"),
                "comment": event.get("comment"),
                "timestamp": event.get("created"),
                "location": event.get("location")
            })

    df = pd.DataFrame(all_events)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', utc=False)
    df = df.sort_values(["sample_name", "timestamp"])
    df["timestamp_end"] = df["timestamp"] + timedelta(days=1)

   
    fig = px.timeline(
        df,
        x_start="timestamp",
        x_end="timestamp_end",
        y="sample_name",
        color="creator",
        hover_data=["creator", "comment", "location"]
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Sample Event Timeline (Durations Between Events)", height=1200)

    fig.show()