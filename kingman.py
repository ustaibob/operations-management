import streamlit as st
import numpy as np

st.title("Kingman's Equation Interactive Visualization")

st.markdown("""
Kingman's equation (VUT equation) estimates the mean waiting time in a single-server queue:

\[
E(W_q) \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \tau
\]

Where:
- \(\\rho\): Utilization \
- \(c_a\): Arrival coefficient of variation \
- \(c_s\): Service coefficient of variation \
- \(\\tau\): Average service time
""")

# User controls
rho = st.slider('Utilization (ρ, λ/μ)', min_value=0.01, max_value=0.99, value=0.7, step=0.01)
ca = st.slider('Arrival Coefficient of Variation (cₐ)', min_value=0.0, max_value=2.0, value=1.0, step=0.01)
cs = st.slider('Service Coefficient of Variation (cₛ)', min_value=0.0, max_value=2.0, value=1.0, step=0.01)
tau = st.slider('Average Service Time (τ)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Kingman's equation calculation
mean_wait = (rho / (1 - rho)) * ((ca**2 + cs**2) / 2) * tau

st.subheader("Result")
st.latex(f"E(W_q) \\approx {mean_wait:.2f}")
st.markdown(f"""
- **Utilization (ρ):** {rho}
- **Arrival Variation (cₐ):** {ca}
- **Service Variation (cₛ):** {cs}
- **Average Service Time (τ):** {tau}
- **Mean Waiting Time:** {mean_wait:.2f}
""")

# Optional: plot how waiting time changes with utilization
import matplotlib.pyplot as plt

rhos = np.linspace(0.01, 0.99, 100)
wait_times = (rhos / (1 - rhos)) * ((ca**2 + cs**2) / 2) * tau

fig, ax = plt.subplots()
ax.plot(rhos, wait_times)
ax.set_xlabel('Utilization (ρ)')
ax.set_ylabel('Mean Waiting Time E(Wq)')
ax.set_title('Waiting Time vs Utilization (fixed ca, cs, τ)')
st.pyplot(fig)
