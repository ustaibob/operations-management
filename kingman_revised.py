import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kingman's Equation", layout="wide")

st.title("Kingman's Equation Interactive Visualization")

st.markdown("""
Kingman's equation (VUT equation) estimates the mean waiting time in a single-server queue:

$$
E(W_q) \\approx \\frac{\\rho}{1-\\rho} \\cdot \\frac{c_a^2 + c_s^2}{2} \\cdot \\tau
$$

**V**ariability × **U**tilization × **T**ime = The fundamental drivers of waiting
""")

# Create two-column layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("System Parameters")

    # User controls with better descriptions
    rho = st.slider(
        'Utilization (ρ = λ/μ)',
        min_value=0.01, max_value=0.99, value=0.7, step=0.01,
        help="Ratio of arrival rate to service rate. Higher values mean the server is busier."
    )

    st.markdown("**Variability:**")
    ca = st.slider(
        'Arrival Coefficient of Variation (cₐ)',
        min_value=0.0, max_value=2.0, value=1.0, step=0.1,
        help="cₐ=0: Regular arrivals, cₐ=1: Poisson (random), cₐ>1: Bursty arrivals"
    )

    cs = st.slider(
        'Service Coefficient of Variation (cₛ)',
        min_value=0.0, max_value=2.0, value=1.0, step=0.1,
        help="cₛ=0: Constant service, cₛ=1: Exponential, cₛ>1: High variance"
    )

    tau = st.slider(
        'Average Service Time (τ)',
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Mean time to serve one customer"
    )

    # Kingman's equation calculation
    mean_wait = (rho / (1 - rho)) * ((ca**2 + cs**2) / 2) * tau

    st.markdown("---")
    st.subheader("Results")

    # Color code based on utilization
    if rho < 0.5:
        status_color = "🟢"
        status = "Low load"
    elif rho < 0.75:
        status_color = "🟡"
        status = "Moderate load"
    elif rho < 0.9:
        status_color = "🟠"
        status = "High load"
    else:
        status_color = "🔴"
        status = "Critical load"

    st.markdown(f"### {status_color} {status}")

    # Display key metrics in metric cards
    st.metric("Mean Waiting Time", f"{mean_wait:.2f} time units")
    st.metric("Total Time in System", f"{mean_wait + tau:.2f} time units")

    variability_factor = (ca**2 + cs**2) / 2
    st.metric("Variability Factor", f"{variability_factor:.2f}")

    # Insights
    st.markdown("---")
    st.markdown("**💡 Key Insights:**")

    if rho > 0.85:
        st.warning("⚠️ High utilization causes exponential growth in wait times!")

    if variability_factor > 1.0:
        st.info(f"📊 High variability is increasing wait time by {variability_factor:.1f}×")
    elif variability_factor < 0.5:
        st.success("✓ Low variability helps reduce waiting time")

with col2:
    st.subheader("Interactive Visualizations")

    # Create a 2x2 grid of plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Waiting time vs Utilization (current variability)
    rhos = np.linspace(0.01, 0.99, 100)
    wait_times = (rhos / (1 - rhos)) * ((ca**2 + cs**2) / 2) * tau

    ax1.plot(rhos, wait_times, 'b-', linewidth=2)
    ax1.axvline(x=rho, color='r', linestyle='--', label=f'Current ρ={rho:.2f}')
    ax1.axhline(y=mean_wait, color='r', linestyle='--', alpha=0.5)
    ax1.fill_between([0, 0.7], 0, max(wait_times), alpha=0.1, color='green', label='Safe zone')
    ax1.fill_between([0.85, 1], 0, max(wait_times), alpha=0.1, color='red', label='Danger zone')
    ax1.set_xlabel('Utilization (ρ)', fontsize=10)
    ax1.set_ylabel('Mean Waiting Time E(Wq)', fontsize=10)
    ax1.set_title('Impact of Utilization', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Plot 2: Waiting time vs Variability (current utilization)
    cvs = np.linspace(0, 2, 100)
    wait_times_ca = (rho / (1 - rho)) * ((cvs**2 + cs**2) / 2) * tau
    wait_times_cs = (rho / (1 - rho)) * ((ca**2 + cvs**2) / 2) * tau

    ax2.plot(cvs, wait_times_ca, 'g-', linewidth=2, label='Varying cₐ')
    ax2.plot(cvs, wait_times_cs, 'm-', linewidth=2, label='Varying cₛ')
    ax2.axvline(x=ca, color='g', linestyle='--', alpha=0.7)
    ax2.axvline(x=cs, color='m', linestyle='--', alpha=0.7)
    ax2.axhline(y=mean_wait, color='r', linestyle='--', alpha=0.5, label='Current wait')
    ax2.set_xlabel('Coefficient of Variation', fontsize=10)
    ax2.set_ylabel('Mean Waiting Time E(Wq)', fontsize=10)
    ax2.set_title('Impact of Variability', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Plot 3: Heatmap of waiting time vs utilization and variability
    rho_range = np.linspace(0.1, 0.95, 50)
    ca_range = np.linspace(0.0, 2.0, 50)
    R, CA = np.meshgrid(rho_range, ca_range)
    W = (R / (1 - R)) * ((CA**2 + cs**2) / 2) * tau

    contour = ax3.contourf(R, CA, W, levels=20, cmap='RdYlGn_r')
    ax3.plot(rho, ca, 'r*', markersize=15, label='Current setting')
    ax3.set_xlabel('Utilization (ρ)', fontsize=10)
    ax3.set_ylabel('Arrival CV (cₐ)', fontsize=10)
    ax3.set_title('Wait Time Heatmap (cₛ fixed)', fontsize=11, fontweight='bold')
    plt.colorbar(contour, ax=ax3, label='E(Wq)')
    ax3.legend(fontsize=8)

    # Plot 4: Component breakdown
    utilization_component = rho / (1 - rho)
    variability_component = (ca**2 + cs**2) / 2

    components = ['Utilization\nFactor', 'Variability\nFactor', 'Service\nTime']
    values = [utilization_component, variability_component, tau]
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    bars = ax4.bar(components, values, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Value', fontsize=10)
    ax4.set_title('VUT Component Breakdown', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Add formula annotation
    ax4.text(0.5, 0.95, f'E(Wq) = {utilization_component:.2f} × {variability_component:.2f} × {tau:.2f} = {mean_wait:.2f}',
             transform=ax4.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# Educational section at bottom
st.markdown("---")
st.subheader("Understanding the VUT Equation")

col_v, col_u, col_t = st.columns(3)

with col_v:
    st.markdown("**V - Variability**")
    st.markdown(f"""
    Factor: `{((ca**2 + cs**2) / 2):.2f}`

    - Captures randomness in arrivals and service
    - Lower variability = more predictable, shorter waits
    - Both arrival and service variation matter
    """)

with col_u:
    st.markdown("**U - Utilization**")
    st.markdown(f"""
    Factor: `{(rho / (1 - rho)):.2f}`

    - Non-linear impact: doubles from {rho:.2f} to {min(0.99, rho + 0.1):.2f}
    - High utilization (>85%) causes congestion
    - Keep below 70-80% for stable performance
    """)

with col_t:
    st.markdown("**T - Time**")
    st.markdown(f"""
    Factor: `{tau:.2f}`

    - Average service time scales waiting linearly
    - Reducing service time directly reduces wait
    - Baseline time unit for the system
    """)
