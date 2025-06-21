"""
Plotly charts and visualizations for the web interface.
"""

import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import streamlit as st
from typing import Dict, Any, List


def create_radar_chart(config) -> go.Figure:
    """Create a radar chart for DDR5 configuration metrics."""
    # Calculate metrics
    bandwidth = (config.frequency * 8 * 2) / 1000  # GB/s
    latency = (config.timings.cl / (config.frequency / 2)) * 1000  # ns
    power_efficiency = bandwidth / (config.voltages.vddq * 1000)  # GB/s per W
    
    # Normalize metrics to 0-100 scale
    bandwidth_score = min(100, (bandwidth / 200) * 100)
    latency_score = max(0, 100 - (latency / 100) * 100)
    power_score = min(100, power_efficiency * 20)
    stability_score = config.get_stability_estimate() if hasattr(config, 'get_stability_estimate') else 85
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[bandwidth_score, latency_score, power_score, stability_score],
        theta=['Bandwidth', 'Latency', 'Power Efficiency', 'Stability'],
        fill='toself',
        name='DDR5 Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="DDR5 Performance Profile"
    )
    
    return fig


def create_optimization_progress_chart(history: List[Dict]) -> go.Figure:
    """Create optimization progress chart."""
    if not history:
        return go.Figure()
    
    history_df = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['performance'],
        mode='lines+markers',
        name='Performance Score',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['stability'],
        mode='lines+markers',
        name='Stability Score',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="AI Optimization Progress",
        xaxis_title="Generation",
        yaxis_title="Score",
        showlegend=True
    )
    
    return fig


def create_comparison_chart(comparison_data: List[Dict]) -> go.Figure:
    """Create frequency comparison chart."""
    df = pd.DataFrame(comparison_data)
    
    fig = px.bar(df, x='Frequency', y='Bandwidth (GB/s)',
                 title='DDR5 Frequency vs Bandwidth',
                 color='Power (mW)',
                 color_continuous_scale='viridis')
    
    return fig


def create_performance_scatter(comparison_data: List[Dict]) -> go.Figure:
    """Create performance vs power scatter plot."""
    df = pd.DataFrame(comparison_data)
    
    fig = px.scatter(df, x='Bandwidth (GB/s)', y='Power (mW)',
                    title='Performance vs Power Consumption',
                    color='Frequency',
                    size='Latency (ns)',
                    hover_data=['Frequency'])
    
    return fig


def create_feature_importance_chart(importance_data: Dict[str, float]) -> go.Figure:
    """Create feature importance bar chart."""
    importance_df = pd.DataFrame(
        list(importance_data.items()),
        columns=['Feature', 'Importance']
    )
    
    fig = px.bar(importance_df, x='Importance', y='Feature',
                orientation='h',
                title='AI Feature Importance Analysis')
    
    return fig


def create_competitor_comparison_chart(competitors: List[Dict]) -> go.Figure:
    """Create competitor comparison chart."""
    competitor_df = pd.DataFrame(competitors)
    
    fig = px.bar(competitor_df, x='name', y='score',
                title='Performance vs Competitors',
                color='score',
                color_continuous_scale='RdYlGn')
    
    return fig
