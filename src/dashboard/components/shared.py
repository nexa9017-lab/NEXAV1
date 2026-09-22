"""
Shared UI components and helper functions for the dashboard.
"""
import streamlit as st

def format_pct(value: float) -> str:
    """Format float as percentage: 0.248 -> 24.8%"""
    return f"{value * 100:.1f}%"

def format_score(value: float) -> str:
    """Format score: 82.36 -> 82.4"""
    return f"{value:.1f}"

def render_prototype_banner():
    """Renders a global notice about the prototype nature of the dashboard."""
    pass

def apply_chart_style(fig):
    """
    Applies consistent styling to Plotly figures.
    """
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="sans-serif", color="#333333"),
        title_font=dict(size=16, color="#111111"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        colorway=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#f0f0f0",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#cccccc"
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#f0f0f0",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#cccccc"
    )
    return fig

def render_error_state(message: str):
    """Renders a standard error state."""
    st.error(f"⚠️ {message}")

def render_empty_state(message: str):
    """Renders a standard empty state."""
    st.info(f"ℹ️ {message}")
