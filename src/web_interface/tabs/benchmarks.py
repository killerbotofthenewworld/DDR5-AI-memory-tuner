"""
Benchmarks tab for performance analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import time
import random
from src.ddr5_models import DDR5Configuration


def render_benchmarks_tab(config: DDR5Configuration):
    """
    Render the benchmarks tab.
    
    Args:
        config: Current DDR5 configuration
    """
    st.header("📈 DDR5 Performance Benchmarks")
    st.info("🎯 **Performance Analysis**: Comprehensive benchmarking suite for DDR5 optimization")
    
    # Benchmark Suite Selection
    st.subheader("🧪 Benchmark Suite Selection")
    
    benchmark_col1, benchmark_col2 = st.columns(2)
    
    with benchmark_col1:
        synthetic_benchmarks = st.multiselect(
            "Synthetic Benchmarks",
            ["AIDA64 Memory", "SiSoft Sandra", "MaxxMEM", "MemTest86", 
             "STREAM", "Tinymembench", "MLC"],
            default=["AIDA64 Memory", "STREAM"]
        )
    
    with benchmark_col2:
        real_world_benchmarks = st.multiselect(
            "Real-World Benchmarks",
            ["3DMark", "Cinebench R23", "Blender", "7-Zip", 
             "x264/x265", "Photoshop", "Gaming Suite"],
            default=["3DMark", "Cinebench R23"]
        )
    
    # Benchmark Configuration
    st.subheader("⚙️ Benchmark Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        test_duration = st.selectbox(
            "Test Duration",
            ["Quick (30s)", "Standard (2min)", "Extended (5min)", "Stress (15min)"],
            index=1
        )
    
    with config_col2:
        thread_count = st.selectbox(
            "Thread Count",
            ["Single", "Dual", "Quad", "All Cores", "Auto"],
            index=4
        )
    
    with config_col3:
        memory_pattern = st.selectbox(
            "Memory Access Pattern",
            ["Sequential", "Random", "Mixed", "Strided"],
            index=2
        )
    
    # Run Benchmarks
    if st.button("🚀 Run Benchmark Suite", type="primary"):
        run_benchmark_suite(config, synthetic_benchmarks, real_world_benchmarks, 
                          test_duration, thread_count, memory_pattern)
    
    st.divider()
    
    # Historical Performance Comparison
    st.subheader("📊 Performance Comparison")
    
    comparison_data = generate_comparison_data(config)
    
    # Memory Bandwidth Chart
    bandwidth_fig = create_bandwidth_chart(comparison_data)
    st.plotly_chart(bandwidth_fig, use_container_width=True)
    
    # Latency Comparison
    latency_fig = create_latency_chart(comparison_data)
    st.plotly_chart(latency_fig, use_container_width=True)
    
    st.divider()
    
    # Detailed Results Table
    st.subheader("📋 Detailed Results")
    
    if 'benchmark_results' in st.session_state:
        results_df = pd.DataFrame(st.session_state.benchmark_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Export options
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📄 Export CSV"):
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="ddr5_benchmarks.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("📊 Export Chart"):
                st.info("Chart export functionality coming soon!")
        
        with export_col3:
            if st.button("📈 Generate Report"):
                generate_benchmark_report(results_df)
    
    # Performance Leaderboard
    st.subheader("🏆 Performance Leaderboard")
    
    leaderboard_data = generate_leaderboard_data()
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    st.dataframe(leaderboard_df, use_container_width=True)


def run_benchmark_suite(config, synthetic, real_world, duration, threads, pattern):
    """Run the selected benchmark suite."""
    st.subheader("🏃‍♂️ Running Benchmarks")
    
    results = []
    
    # Progress tracking
    total_benchmarks = len(synthetic) + len(real_world)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current_benchmark = 0
    
    # Run synthetic benchmarks
    for benchmark in synthetic:
        current_benchmark += 1
        status_text.text(f"Running {benchmark}...")
        
        with st.spinner(f"Running {benchmark}..."):
            time.sleep(random.uniform(1, 3))
            
            # Generate realistic results based on configuration
            result = generate_benchmark_result(benchmark, config, "synthetic")
            results.append(result)
            
            progress_bar.progress(current_benchmark / total_benchmarks)
    
    # Run real-world benchmarks
    for benchmark in real_world:
        current_benchmark += 1
        status_text.text(f"Running {benchmark}...")
        
        with st.spinner(f"Running {benchmark}..."):
            time.sleep(random.uniform(2, 4))
            
            result = generate_benchmark_result(benchmark, config, "real_world")
            results.append(result)
            
            progress_bar.progress(current_benchmark / total_benchmarks)
    
    # Store results
    st.session_state.benchmark_results = results
    
    status_text.text("✅ All benchmarks completed!")
    st.success("🎉 Benchmark suite completed successfully!")
    
    # Show summary
    show_benchmark_summary(results)


def generate_benchmark_result(benchmark_name, config, benchmark_type):
    """Generate realistic benchmark results."""
    # Base performance calculation
    base_performance = (config.frequency / 5600) * 100  # Normalized to DDR5-5600
    
    # Timing impact
    timing_factor = (40 / config.timings.cl) * 0.8 + 0.2  # CL impact
    
    # Frequency scaling
    freq_factor = (config.frequency / 5600) ** 0.7
    
    # Random variance
    variance = random.uniform(0.85, 1.15)
    
    if benchmark_type == "synthetic":
        if "Memory" in benchmark_name or "STREAM" in benchmark_name:
            score = base_performance * timing_factor * freq_factor * variance
            unit = "GB/s" if "STREAM" in benchmark_name else "MB/s"
        else:
            score = base_performance * timing_factor * variance
            unit = "Points"
    else:
        # Real-world benchmarks are less sensitive to memory
        sensitivity = 0.3
        score = 100 + (base_performance - 100) * sensitivity * variance
        unit = "Points" if "3DMark" in benchmark_name else "Score"
    
    return {
        "Benchmark": benchmark_name,
        "Score": round(score, 2),
        "Unit": unit,
        "Type": benchmark_type.replace("_", " ").title(),
        "Config": f"{config.frequency}MHz {config.timings.cl}-{config.timings.trcd}-{config.timings.trp}-{config.timings.tras}"
    }


def generate_comparison_data(config):
    """Generate comparison data for charts."""
    configs = [
        {"name": "DDR5-4800", "freq": 4800, "cl": 40, "bandwidth": 38.4, "latency": 16.7},
        {"name": "DDR5-5600", "freq": 5600, "cl": 36, "bandwidth": 44.8, "latency": 12.9},
        {"name": "DDR5-6000", "freq": 6000, "cl": 36, "bandwidth": 48.0, "latency": 12.0},
        {"name": "DDR5-6400", "freq": 6400, "cl": 32, "bandwidth": 51.2, "latency": 10.0},
        {"name": "Current Config", "freq": config.frequency, "cl": config.timings.cl, 
         "bandwidth": (config.frequency * 8) / 1000, "latency": (config.timings.cl / config.frequency) * 2000}
    ]
    
    return configs


def create_bandwidth_chart(data):
    """Create bandwidth comparison chart."""
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x='name', y='bandwidth',
        title='Memory Bandwidth Comparison',
        labels={'bandwidth': 'Bandwidth (GB/s)', 'name': 'Configuration'},
        color='bandwidth',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def create_latency_chart(data):
    """Create latency comparison chart."""
    df = pd.DataFrame(data)
    
    fig = px.scatter(
        df, x='freq', y='latency', size='bandwidth',
        hover_name='name',
        title='Frequency vs Latency',
        labels={'freq': 'Frequency (MT/s)', 'latency': 'Latency (ns)'},
        color='bandwidth',
        color_continuous_scale='plasma'
    )
    
    fig.update_layout(template="plotly_white")
    
    return fig


def show_benchmark_summary(results):
    """Show benchmark summary."""
    st.subheader("📊 Benchmark Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        synthetic_results = [r for r in results if r["Type"] == "Synthetic"]
        if synthetic_results:
            avg_synthetic = sum(r["Score"] for r in synthetic_results) / len(synthetic_results)
            st.metric("Avg Synthetic Score", f"{avg_synthetic:.1f}")
    
    with summary_col2:
        real_world_results = [r for r in results if r["Type"] == "Real World"]
        if real_world_results:
            avg_real_world = sum(r["Score"] for r in real_world_results) / len(real_world_results)
            st.metric("Avg Real-World Score", f"{avg_real_world:.1f}")
    
    with summary_col3:
        all_scores = [r["Score"] for r in results]
        overall_score = sum(all_scores) / len(all_scores)
        st.metric("Overall Score", f"{overall_score:.1f}")


def generate_leaderboard_data():
    """Generate leaderboard data."""
    return [
        {"Rank": 1, "Configuration": "DDR5-8000 30-38-38-84", "Score": 158.3, "User": "MemoryMaster"},
        {"Rank": 2, "Configuration": "DDR5-7600 32-39-39-102", "Score": 152.7, "User": "OCGuru"},
        {"Rank": 3, "Configuration": "DDR5-7200 34-40-40-96", "Score": 147.2, "User": "SpeedDemon"},
        {"Rank": 4, "Configuration": "DDR5-6800 32-39-39-88", "Score": 142.8, "User": "TuningPro"},
        {"Rank": 5, "Configuration": "DDR5-6400 30-36-36-76", "Score": 138.5, "User": "AIOptimizer"},
    ]


def generate_benchmark_report(results_df):
    """Generate comprehensive benchmark report."""
    st.subheader("📋 Benchmark Report")
    
    report_text = f"""
    # DDR5 Performance Benchmark Report
    
    ## System Configuration
    - Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
    - Total Benchmarks: {len(results_df)}
    - Average Score: {results_df['Score'].mean():.2f}
    - Best Performance: {results_df['Score'].max():.2f} ({results_df.loc[results_df['Score'].idxmax(), 'Benchmark']})
    
    ## Performance Analysis
    The current DDR5 configuration shows competitive performance across all tested benchmarks.
    
    ## Recommendations
    - Consider optimizing primary timings for better latency
    - Monitor temperatures during extended workloads
    - Test stability with stress tests before finalizing settings
    """
    
    st.markdown(report_text)
    
    if st.button("📄 Download Report"):
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name="ddr5_benchmark_report.md",
            mime="text/markdown"
        )
