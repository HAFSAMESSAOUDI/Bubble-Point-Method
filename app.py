import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import streamlit as st
import os

# ------------------------- Helper Functions -------------------------

def load_k_values_from_csv(component):
    filepath = f"data/{component}.csv"
    if not os.path.exists(filepath):
        st.error(f"Le fichier {filepath} est introuvable. Vérifiez qu'il est inclus dans le dépôt.")
        st.stop()

    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()  # Supprime les espaces autour des noms de colonnes

    if "T" not in data.columns or "K" not in data.columns:
        st.error(f"Le fichier {component}.csv doit contenir des colonnes 'T' et 'K'.")
        st.stop()

    return interp1d(data["T"], data["K"], kind="linear", fill_value="extrapolate")

def calculate_corrected_coefficients(N, T_dict, F_dict, z_i_dict, V_dict, U_dict, K_values):
    A = np.zeros(N - 1)
    B = np.zeros(N)
    C = np.zeros(N - 1)
    D = np.zeros(N)

    for j in range(1, N + 1):
        idx = j - 1
        K_ij = K_values[idx]
        F_j = F_dict.get(j, 0.0)
        z_ij = z_i_dict.get(j, 0.0)
        D[idx] = -F_j * z_ij
        if j > 1:
            sum_Fm_Um = sum(F_dict.get(m, 0.0) - U_dict.get(m, 0.0) for m in range(1, j))
            A[idx - 1] = V_dict[j] + sum_Fm_Um
        V_jp1 = V_dict[j + 1] if j < N else 0.0
        sum_Fm_Um_B = sum(F_dict.get(m, 0.0) - U_dict.get(m, 0.0) for m in range(1, j + 1))
        B[idx] = -(V_jp1 + sum_Fm_Um_B + U_dict[j] + V_dict[j] * K_ij)
        if j < N:
            K_ijp1 = K_values[idx + 1]
            C[idx] = V_dict[j + 1] * K_ijp1

    return A, B, C, D

def thomas_algorithm(A, B, C, D):
    N = len(B)
    P = np.zeros(N - 1)
    Q = np.zeros(N)
    P[0] = C[0] / B[0]
    Q[0] = D[0] / B[0]

    for i in range(1, N):
        denominator = B[i] - A[i - 1] * P[i - 1]
        if i < N - 1:
            P[i] = C[i] / denominator
        Q[i] = (D[i] - A[i - 1] * Q[i - 1]) / denominator

    x = np.zeros(N)
    x[-1] = Q[-1]
    for i in range(N - 2, -1, -1):
        x[i] = Q[i] - P[i] * x[i + 1]
    return x

# ------------------------- Main Simulation Function -------------------------

def run_simulation(max_iterations, tolerance):
    N = 5
    T_dict = {1: 65.0, 2: 90.0, 3: 115.0, 4: 140.0, 5: 165.0}
    F_dict = {1: 0.0, 2: 0.0, 3: 100.0, 4: 0.0, 5: 0.0}
    z_C3 = {3: 0.30}
    z_nC4 = {3: 0.30}
    z_nC5 = {3: 0.40}
    V_dict = {1: 0.0, 2: 150.0, 3: 150.0, 4: 150.0, 5: 150.0}
    U_dict = {1: 50.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    components = ["C3", "nC4", "nC5"]
    C = len(components)

    results = {}
    x_normalized = {comp: np.zeros(N) for comp in components}
    stage_sums = np.zeros(N)

    output_logs = []

    for iteration in range(max_iterations):
        stage_sums.fill(0)
        S_values = []
        st.subheader(f"Iteration {iteration + 1}")
        cols = st.columns(len(components))
        for i, comp in enumerate(components):
            k_interp = load_k_values_from_csv(comp)
            K_values = [k_interp(T_dict[j]) for j in range(1, N + 1)]
            z_dict = locals()[f"z_{comp}"]
            A_calc, B_calc, C_calc, D_calc = calculate_corrected_coefficients(
                N, T_dict, F_dict, z_dict, V_dict, U_dict, K_values
            )
            solution = thomas_algorithm(A_calc, B_calc, C_calc, D_calc)

            coeff_df = pd.DataFrame({
                "A": np.pad(A_calc, (1, 0), 'constant'),
                "B": B_calc,
                "C": np.pad(C_calc, (0, 1), 'constant'),
                "D": D_calc
            })

            with cols[i]:
                st.subheader(f"{comp} Coefficients")
                st.dataframe(coeff_df)

            stage_sums += solution
            results[comp] = solution

        for comp in components:
            for j in range(N):
                x_normalized[comp][j] = results[comp][j] / (stage_sums[j] / C)

        new_T_dict = {}
        for j in range(1, N + 1):
            sum_Kx = sum(
                load_k_values_from_csv(comp)(T_dict[j]) * x_normalized[comp][j - 1]
                for comp in components
            )
            S_j = sum_Kx - 1
            S_values.append(S_j)
            new_T_dict[j] = T_dict[j] - S_j * 0.1

        max_temp_diff = max(abs(new_T_dict[j] - T_dict[j]) for j in range(1, N + 1))
        st.text(f"S_j values: {', '.join(f'{S:.4f}' for S in S_values)}")

        if max_temp_diff < tolerance:
            st.success("Converged!")
            break
        T_dict = new_T_dict

    final_results = {
        "logs": output_logs,
        "x_normalized": {comp: list(map(float, x_normalized[comp])) for comp in components},
        "stage_temperatures": [round(T_dict[j], 2) for j in range(1, N + 1)],
        "S_values": S_values,
    }
    return final_results

# ------------------------- Streamlit Interface -------------------------

st.title("Distillation Simulation")
st.sidebar.header("Simulation Parameters")

max_iterations = st.sidebar.number_input("Max Iterations", min_value=1, value=12)
tolerance = st.sidebar.number_input("Tolerance", min_value=0.0001, value=0.01, step=0.0001)

if st.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        results = run_simulation(max_iterations, tolerance)

    st.subheader("Normalized x_ij")
    normalized_df = pd.DataFrame(results["x_normalized"])
    st.dataframe(normalized_df)

    st.subheader("Stage Temperatures")
    stage_temp_df = pd.DataFrame({"Stage": range(1, len(results["stage_temperatures"]) + 1), "Temperature (°F)": results["stage_temperatures"]})
    st.dataframe(stage_temp_df)
