"""
Enhanced UI styling with dark/light theme support
"""
import streamlit as st

def load_custom_css():
    """Load custom CSS with dark/light theme support"""
    
    # Get current theme preference
    theme = st.session_state.get('theme', 'dark')
    
    # Define color schemes
    themes = {
        'dark': {
            'bg_primary': '#0e1117',
            'bg_secondary': '#262730',
            'bg_tertiary': '#1e2028',
            'text_primary': '#fafafa',
            'text_secondary': '#a6a6a6',
            'accent': '#ff6b35',
            'success': '#00d084',
            'warning': '#ffb800',
            'error': '#ff4757',
            'border': '#464852'
        },
        'light': {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8f9fa',
            'bg_tertiary': '#e9ecef',
            'text_primary': '#212529',
            'text_secondary': '#6c757d',
            'accent': '#007bff',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'border': '#dee2e6'
        }
    }
    
    colors = themes[theme]
    
    custom_css = f"""
    <style>
    /* Main theme variables */
    :root {{
        --bg-primary: {colors['bg_primary']};
        --bg-secondary: {colors['bg_secondary']};
        --bg-tertiary: {colors['bg_tertiary']};
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --accent: {colors['accent']};
        --success: {colors['success']};
        --warning: {colors['warning']};
        --error: {colors['error']};
        --border: {colors['border']};
    }}
    
    /* Streamlit app styling */
    .stApp {{
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background: var(--bg-tertiary);
        border-right: 2px solid var(--border);
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(90deg, var(--accent) 0%, var(--success) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    
    /* Card styling */
    .metric-card {{
        background: var(--bg-tertiary);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid var(--border);
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(45deg, var(--accent), var(--success));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }}
    
    /* Progress bar styling */
    .progress-container {{
        background: var(--bg-secondary);
        border-radius: 25px;
        overflow: hidden;
        height: 30px;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .progress-bar {{
        height: 100%;
        background: linear-gradient(90deg, var(--accent), var(--success));
        border-radius: 25px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        position: relative;
        overflow: hidden;
    }}
    
    .progress-bar::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}
    
    /* Loading animation */
    .loading-spinner {{
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid var(--border);
        border-radius: 50%;
        border-top-color: var(--accent);
        animation: spin 1s ease-in-out infinite;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    
    /* Pulse animation */
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
        100% {{ opacity: 1; }}
    }}
    
    /* Alert styling */
    .alert {{
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }}
    
    .alert-success {{
        background: rgba(40, 167, 69, 0.1);
        border-left-color: var(--success);
        color: var(--success);
    }}
    
    .alert-warning {{
        background: rgba(255, 193, 7, 0.1);
        border-left-color: var(--warning);
        color: var(--warning);
    }}
    
    .alert-error {{
        background: rgba(220, 53, 69, 0.1);
        border-left-color: var(--error);
        color: var(--error);
    }}
    
    /* Theme toggle button */
    .theme-toggle {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        cursor: pointer;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    .theme-toggle:hover {{
        transform: scale(1.1);
    }}
    
    /* Chart containers */
    .chart-container {{
        background: var(--bg-tertiary);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid var(--border);
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .metric-card {{
            padding: 1rem;
        }}
        
        .main-header {{
            padding: 0.5rem;
        }}
    }}
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def theme_toggle():
    """Theme toggle component"""
    current_theme = st.session_state.get('theme', 'dark')
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🌓 Toggle Theme", key="theme_toggle"):
            st.session_state['theme'] = 'light' if current_theme == 'dark' else 'dark'
            st.rerun()

def create_progress_bar(value: float, max_value: float = 100, label: str = "Progress"):
    """Create animated progress bar"""
    percentage = min((value / max_value) * 100, 100)
    
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {percentage}%;">
            {label}: {percentage:.1f}%
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

def loading_spinner(text: str = "Loading..."):
    """Create loading spinner"""
    spinner_html = f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <p style="margin-top: 1rem;">{text}</p>
    </div>
    """
    
    st.markdown(spinner_html, unsafe_allow_html=True)

def create_alert(message: str, alert_type: str = "success"):
    """Create styled alert"""
    alert_html = f"""
    <div class="alert alert-{alert_type}">
        {message}
    </div>
    """
    
    st.markdown(alert_html, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, description: str = "", delta: str = ""):
    """Create styled metric card"""
    delta_html = f"<small style='color: var(--success);'>{delta}</small>" if delta else ""
    
    card_html = f"""
    <div class="metric-card">
        <h3 style="margin: 0 0 0.5rem 0; color: var(--accent);">{title}</h3>
        <h2 style="margin: 0; color: var(--text-primary);">{value} {delta_html}</h2>
        <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">{description}</p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
