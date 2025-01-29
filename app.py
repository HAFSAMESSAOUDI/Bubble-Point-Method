from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import gradio as gr
import seaborn as sns
import os
import groq



# ------------------------- Helper Functions -------------------------

def load_k_values_from_csv(component):
    filepath = f"data/{component}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable. Vérifiez qu'il est inclus dans le dépôt.")
    
    # Charger le fichier CSV
    data = pd.read_csv(filepath)

    # Nettoyer les noms de colonnes pour supprimer les espaces ou caractères invisibles
    data.columns = data.columns.str.strip()

    # Vérifiez que les colonnes 'T' et 'K' existent
    if "T" not in data.columns or "K" not in data.columns:
        raise ValueError(f"Le fichier {component}.csv doit contenir des colonnes 'T' et 'K'.")
    
    # Créer l'interpolateur
    return interp1d(data["T"], data["K"], kind="linear", fill_value="extrapolate")


def calculate_corrected_coefficients(N, T_dict, F_dict, z_i_dict, V_dict, U_dict, K_values):
    A = np.zeros(N - 1)  # Subdiagonal values
    B = np.zeros(N)      # Diagonal values
    C = np.zeros(N - 1)  # Superdiagonal values
    D = np.zeros(N)      # Right-hand side vector

    for j in range(1, N + 1):
        idx = j - 1
        K_ij = K_values[idx]
        F_j = F_dict.get(j, 0.0)
        z_ij = z_i_dict.get(j, 0.0)
        D[idx] = -F_j * z_ij  # Compute D values
        
        if j > 1:
            sum_Fm_Um = sum(F_dict.get(m, 0.0) - U_dict.get(m, 0.0) for m in range(1, j))
            A[idx - 1] = V_dict[j] + sum_Fm_Um  # Subdiagonal (A)

        V_jp1 = V_dict[j + 1] if j < N else 0.0
        sum_Fm_Um_B = sum(F_dict.get(m, 0.0) - U_dict.get(m, 0.0) for m in range(1, j + 1))
        B[idx] = -(V_jp1 + sum_Fm_Um_B + U_dict[j] + V_dict[j] * K_ij)  # Diagonal (B)

        if j < N:
            K_ijp1 = K_values[idx + 1]
            C[idx] = V_dict[j + 1] * K_ijp1  # Superdiagonal (C)

    # Construct the tridiagonal matrix
    tridiagonal_matrix = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            tridiagonal_matrix[i][i - 1] = A[i - 1]  # Subdiagonal
        tridiagonal_matrix[i][i] = B[i]            # Diagonal
        if i < N - 1:
            tridiagonal_matrix[i][i + 1] = C[i]    # Superdiagonal

    D = np.where(np.abs(D) < 1e-10, 0, D)  # Remove small numerical artifacts in D
    return A, B, C, D, tridiagonal_matrix


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

    results = {}
    x_normalized = {comp: np.zeros(N) for comp in components}
    stage_sums = np.zeros(N)

    output_logs = []
    S_all_iterations = []

    for iteration in range(max_iterations):
        stage_sums.fill(0)
        S_values = []
        iteration_tables = []

        for comp in components:
            k_interp = load_k_values_from_csv(comp)
            K_values = [k_interp(T_dict[j]) for j in range(1, N + 1)]
            z_dict = locals()[f"z_{comp}"]
            A_calc, B_calc, C_calc, D_calc, tridiagonal_matrix = calculate_corrected_coefficients(
                N, T_dict, F_dict, z_dict, V_dict, U_dict, K_values
            )
            solution = thomas_algorithm(A_calc, B_calc, C_calc, D_calc)

            # Save the tridiagonal matrix as a DataFrame for display
            df_tridiagonal_with_D = pd.DataFrame(
                tridiagonal_matrix,
                columns=[f"Col {i + 1}" for i in range(N)],
                index=[f"Row {i + 1}" for i in range(N)]
            )       
            df_tridiagonal_with_D["D"] = D_calc 
            iteration_tables.append((comp, df_tridiagonal_with_D, pd.Series(solution, name="x_{i,j}")))

            stage_sums += solution
            results[comp] = solution

        for comp in components:
            for j in range(N):
                x_normalized[comp][j] = results[comp][j] / (stage_sums[j] / len(components))

        # Compute new temperatures and S_j values
        new_T_dict = {}
        for j in range(1, N + 1):
            sum_Kx = sum(
                load_k_values_from_csv(comp)(T_dict[j]) * x_normalized[comp][j - 1]
                for comp in components
            )
            S_j = sum_Kx - 1
            S_values.append(S_j)
            new_T_dict[j] = T_dict[j] - S_j * 0.1

        S_all_iterations.append(S_values)
        max_temp_diff = max(abs(new_T_dict[j] - T_dict[j]) for j in range(1, N + 1))

        # Store iteration data
        output_logs.append({
            "Iteration": iteration + 1,
            "Tables": iteration_tables,
            "S_j": pd.DataFrame({"Stage": [f"Stage {j}" for j in range(1, N + 1)], "S_j": S_values}),
        })

        if max_temp_diff < tolerance:
            break

        T_dict = new_T_dict

    final_results = {
        "logs": output_logs,
        "x_normalized": {comp: list(map(float, x_normalized[comp])) for comp in components},
        "stage_temperatures": [round(T_dict[j], 2) for j in range(1, N + 1)],
        "S_values": S_all_iterations,
    }
    return final_results


def save_results_to_csv(results):
    df = pd.DataFrame(results["x_normalized"])
    df["Stage Temperatures"] = results["stage_temperatures"]
    filepath = "simulation_results.csv"
    df.to_csv(filepath, index=False)
    return filepath

# ------------------------- Gradio Interface -------------------------

custom_css = """
table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
}

th, td {
    border: 1px solid black;
    text-align: center;
    padding: 8px;
}

th {
    background-color: #007bff;
    color: black;
}

tr:nth-child(even) {
    background-color:rgb(0, 0, 0);
}
# Style les colonnes de composants en bleu
def style_results(df):
    return df.style.set_properties(
        subset=["C3", "nC4", "nC5"],
        **{"color": "blue", "font-weight": "bold"}
    )

"""


def gradio_interface(max_iterations, tolerance, plot_choice):
    results = run_simulation(int(max_iterations), float(tolerance))
    logs = results["logs"]
    x_normalized = results["x_normalized"]
    stage_temperatures = results["stage_temperatures"]
    S_values = results["S_values"]
    

    # Préparation des tableaux formatés pour chaque itération
    formatted_logs = ""
    for log in logs:
        formatted_logs += f"<h3>Iteration {log['Iteration']}</h3>"
        formatted_logs += "<div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>"

        # Ajout des matrices pour chaque composant
        for comp, df_tridiagonal_with_D, solution in log["Tables"]:
            formatted_logs += (
                f"<div style='margin: 10px; text-align: center;'>"
                f"<h4>{comp} Tridiagonal Matrix (with D)</h4>"
                f"{df_tridiagonal_with_D.to_html(index=True, justify='center', border=1)}"
                f"<h4>{comp} Solution x_{{i,j}}:</h4>"
                f"{pd.DataFrame({'x_{i,j}': solution}).to_html(index=False, justify='center', border=1)}"
                f"</div>"
            )

        formatted_logs += "</div>"
        formatted_logs += f"<h4>S_j Values:</h4>{log['S_j'].to_html(index=False, justify='center', border=1)}<hr>"

    # Création du tableau des résultats finaux (normalisés)
    df_results = pd.DataFrame(x_normalized)
    df_results["Stage Temperatures"] = stage_temperatures
    

    # Enregistrer les résultats au format CSV pour téléchargement
    filepath = save_results_to_csv(results)

    # Génération des graphiques en fonction du choix
    plot = None
    if plot_choice == "Évolution des S_j au cours des itérations":
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for iteration, S_vals in enumerate(S_values):
            plt.plot(range(1, len(S_vals) + 1), S_vals, marker="o", label=f"Iteration {iteration + 1}")
        plt.xlabel("Étages")
        plt.ylabel("Valeurs \(S_j\)")
        plt.title("Évolution des \(S_j\) au cours des itérations")
        plt.legend()
        plot = gr.Plot(plt.gcf())
        plt.close()

    elif plot_choice == "Distribution des x_i,j par étage":
        import matplotlib.pyplot as plt
        stages = list(range(1, len(next(iter(x_normalized.values()))) + 1))
        plt.figure(figsize=(10, 6))
        bar_width = 0.2
        positions = np.arange(len(stages))

        for idx, (comp, values) in enumerate(x_normalized.items()):
            plt.bar(positions + idx * bar_width, values, width=bar_width, label=comp)

        plt.xlabel("Étages")
        plt.ylabel("Valeur de \(x_{i,j}\)")
        plt.title("Distribution des \(x_{i,j}\) par étage")
        plt.xticks(positions + bar_width * (len(x_normalized) - 1) / 2, stages)
        plt.legend(title="Composants")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plot = gr.Plot(plt.gcf())
        plt.close()

    elif plot_choice == "Heatmap de la Matrice Tridiagonale":
        import seaborn as sns
        import matplotlib.pyplot as plt

        last_iteration_matrices = logs[-1]["Tables"]
        num_components = len(last_iteration_matrices)

        fig, axes = plt.subplots(1, num_components, figsize=(6 * num_components, 5), constrained_layout=True)

        if num_components == 1:
            axes = [axes]

        for ax, (comp, df_tridiagonal_with_D, _) in zip(axes, last_iteration_matrices):
            sns.heatmap(
                df_tridiagonal_with_D.iloc[:, :-1],
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax
            )
            ax.set_title(f"Heatmap: Matrice Tridiagonale - {comp}")
            ax.set_xlabel("Colonnes")
            ax.set_ylabel("Lignes")

        plot = gr.Plot(plt.gcf())
        plt.close()

    elif plot_choice == "Variation de la température en fonction des étages":
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(stage_temperatures) + 1), stage_temperatures, marker="o", color="blue")
        plt.xlabel("Étages")
        plt.ylabel("Température (°C)")
        plt.title("Variation de la température en fonction des étages")
        plt.grid(axis="both", linestyle="--", alpha=0.7)
        plot = gr.Plot(plt.gcf())
        plt.close()

    return formatted_logs, df_results, filepath, plot
# Configurer l'API Groq
client = groq.Client(api_key="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")  # Mets ta vraie clé ici

def chatgpt_response(prompt, history=[]):
    """Gère les requêtes à Groq avec gestion des erreurs"""
    if history is None:
        history = []

    messages = [{"role": "system", "content": "Tu es un assistant expert en simulation de distillation."}]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur API Groq : {e}"
# Gradio Interface Setup
iface = gr.Blocks(css=custom_css)

with iface:
    # Add a centered title to the app
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #007bff; font-family: Arial, sans-serif;">
                Simulation of Tridiagonal Matrices and Solutions
            </h1>
            <p>Explore the iterative simulation of components, view detailed matrices, and analyze results.</p>
        </div>
        """
    )
    
    # Input Row: Parameters for the simulation
    with gr.Row():
        max_iterations = gr.Number(label="Max Iterations", value=12)
        tolerance = gr.Number(label="Tolerance", value=0.01)

    # Dropdown to select plot type
    with gr.Row():
        plot_choice = gr.Dropdown(
            label="Choisissez le type de graphique à afficher",
            choices=[
                "Évolution des S_j au cours des itérations",
                "Distribution des x_i,j par étage",
                "Heatmap de la Matrice Tridiagonale",
                "Variation de la température en fonction des étages"
            ],
            value="Évolution des S_j au cours des itérations"
        )

    # Display detailed iteration logs and component results
    with gr.Row():
        logs_summary = gr.HTML(label="Logs and Detailed Iteration Results")

    # Unified table for final normalized results and stage temperatures
    with gr.Row():
        results_table = gr.DataFrame(label="Normalized Results and Temperatures")
        
    # Download link for results
    with gr.Row():
        download_file = gr.File(label="Download Results")

    # Placeholder for plot output
    with gr.Row():
        plot_output = gr.Plot(label="Visualisation du graphique")

    # Submit Button
    submit_button = gr.Button("Submit")
    submit_button.click(
        fn=gradio_interface,
        inputs=[max_iterations, tolerance, plot_choice],
        outputs=[logs_summary, results_table, download_file, plot_output],
    )
# Chatbot Groq
    with gr.Row():
        chatbot = gr.Chatbot(label="Assistant IA", height=400)

    with gr.Row():
        chat_input = gr.Textbox(label="Posez une question à ChatGPT", placeholder="Tapez votre question ici...")
        send_button = gr.Button("Envoyer")

    def chat_interaction(message, history):
        response = chatgpt_response(message, history)
        history.append((message, response))
        return history, ""

    send_button.click(chat_interaction, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
# Lancer l'interface
iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))