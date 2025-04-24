import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
import math
# Optional: Use sklearn for metrics, or calculate manually
# from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
EXCEL_FILE_PATH = '/content/stability_data.xlsx' # <--- Make sure this file exists!

# !!! IMPORTANT: Adjust HMWP_MAX based on expected plateau !!!
# This value significantly affects the calculated 'y' and model curvature.
# If curves look too flat or fitting fails, systematically try different values.
# Common values to try: 100.0, 50.0, 20.0, 10.0, 5.0
HMWP_MAX = 5.0 # Assumed maximum HMWP (%), adjusted based on previous results

# --- Weighting Configuration ---
# Apply higher weight (lower sigma) to 5C data to improve its fit
apply_weighting = False # Set to False to disable weighting (User preference)
weight_factor_5C = 5 # (Not used if apply_weighting is False) How much more weight for 5C data

TIME_CUTOFF_DAYS = 97 # Used for visual distinction in plot only in global fit
PREDICTION_YEARS = 3 # How many years out to predict

# --- Constants ---
R = 8.314e-3  # Gas constant in kJ/mol/K

# --- 1. Data Loading and Preparation ---
print(f"--- Loading Data from {EXCEL_FILE_PATH} ---")
try:
    df = pd.read_excel(EXCEL_FILE_PATH)
    # Basic validation
    required_cols = ['Experiment', 'Temperature_C', 'Time_days', 'HMWP']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Excel file must contain columns: {required_cols}")
    # Drop rows with missing values in essential columns
    df.dropna(subset=required_cols, inplace=True)
    if df.empty:
        raise ValueError("No valid data remaining after dropping rows with missing values.")
    print(f"  Loaded {len(df)} valid data points from {df['Experiment'].nunique()} experiments.")
except FileNotFoundError:
    print(f"ERROR: Excel file not found at '{EXCEL_FILE_PATH}'. Please create it.")
    exit()
except ValueError as e:
    print(f"ERROR: {e}")
    exit()
except Exception as e:
    print(f"ERROR loading or validating Excel file: {e}")
    exit()

# Convert Celsius to Kelvin
df['Temperature_K'] = df['Temperature_C'] + 273.15

# Define fractional conversion 'y' using the configured HMWP_MAX
df['y'] = df['HMWP'] / HMWP_MAX
# Ensure y doesn't exceed 1 due to potential noise slightly above HMWP_MAX
df['y'] = df['y'].clip(upper=1.0)

# Get unique batch names (Experiments)
unique_batches = df['Experiment'].unique()
batch_map = {name: i for i, name in enumerate(unique_batches)} # Map name to index

# Determine initial condition y0 for each batch
y0_per_batch = {}
for batch_name in unique_batches:
    batch_data = df[df['Experiment'] == batch_name]
    # Find t=0 data, considering potential floating point inaccuracies
    t0_data = batch_data[np.isclose(batch_data['Time_days'], 0)]
    if t0_data.empty:
        min_y_in_batch = batch_data['y'].min()
        warnings.warn(f"Warning: No t=0 data found for batch '{batch_name}'. Using minimum observed y ({min_y_in_batch:.4f}) as y0.")
        y0_per_batch[batch_name] = min_y_in_batch
    else:
        # Handle potential multiple t=0 entries (e.g., different temps) - use the mean y at t=0
        y0_val = t0_data['y'].mean()
        # Ensure y0 is not greater than 1 (can happen if HMWP_MAX is small and initial HMWP is high)
        y0_per_batch[batch_name] = min(y0_val, 1.0 - 1e-9) # Clip y0 just below 1 if needed
print(f"  Initial conditions (y0) determined for {len(y0_per_batch)} batches.")
print(f"  Using HMWP_MAX = {HMWP_MAX} for calculating fractional conversion 'y'.")

# --- 2. Define the Differential Equation Model ---
# (model_ode function remains the same as before)
def model_ode(t, y, k, n, m):
    """
    Differential equation dy/dt = k * y^n * (1-y)^m
    Includes clamping to avoid numerical issues near y=0 or y=1.
    """
    eps = 1e-9
    # Clamp y slightly away from 0 and 1 before applying exponents
    # Ensure y is treated as an array/float for numpy functions
    y_val = y[0] if isinstance(y, (list, np.ndarray)) else y
    y_clamped = np.maximum(eps, np.minimum(y_val, 1.0 - eps))

    # Ensure bases for powers are non-negative
    term1_base = y_clamped
    term2_base = 1.0 - y_clamped

    # Calculate terms safely
    term1 = term1_base**n if term1_base >= 0 else 0
    term2 = term2_base**m if term2_base >= 0 else 0

    dydt = k * term1 * term2
    # Ensure rate is non-negative
    return max(0, dydt)

# --- 3. Define the Global Fitting Function (Multi-Batch) ---
# (global_model_simulate_multibatch function remains the same as before)
def global_model_simulate_multibatch(df_input_indices, Ea, n, m, *logAs_batch):
    """
    Simulates HMWP formation for all experiments based on index.
    df_input_indices: Indices corresponding to rows in the global DataFrame 'df'.
    Ea, n, m: Global kinetic parameters.
    *logAs_batch: Fitted log(A) values, one for each batch in order.
    """
    df_subset = df.loc[df_input_indices]
    y_predicted_all = np.zeros(len(df_subset), dtype=float)
    logA_dict = {batch_name: logA for batch_name, logA in zip(unique_batches, logAs_batch)}
    eps = 1e-9 # Small epsilon for comparisons and bounds

    for i, index in enumerate(df_input_indices):
        row = df_subset.loc[index]
        batch_name = row['Experiment']
        temp_K = row['Temperature_K']
        time_point = row['Time_days']

        logA = logA_dict[batch_name]
        y0 = y0_per_batch[batch_name]
        A = np.exp(logA)
        k = A * np.exp(-Ea / (R * temp_K))

        # Ensure k is positive and finite
        if not np.isfinite(k) or k <= 0:
             warnings.warn(f"Invalid rate constant k={k} calculated for {batch_name}, T={temp_K-273.15:.1f}°C. Parameters: Ea={Ea:.2f}, logA={logA:.2f}")
             return np.full(len(df_subset), 1e10) # Return large penalty

        if np.isclose(time_point, 0):
            y_pred = y0
        else:
            try:
                sol = solve_ivp(
                    model_ode,
                    t_span=[0, time_point],
                    y0=[y0], # solve_ivp expects y0 as a list or array
                    t_eval=[time_point],
                    args=(k, n, m),
                    method='RK45',
                    rtol=1e-6, atol=1e-8
                )

                if sol.status != 0:
                    # Don't warn every time, can flood console. Return penalty.
                    # warnings.warn(f"ODE solver failed for {batch_name}, T={temp_K-273.15:.1f}°C, t={time_point}d. Status: {sol.status}")
                    return np.full(len(df_subset), 1e10) # Return large penalty

                y_pred = sol.y[0, -1]

            except Exception as ode_err:
                # Catch potential errors during ODE solving (e.g., from bad parameters n, m)
                # warnings.warn(f"ODE solver error for {batch_name}, T={temp_K-273.15:.1f}°C, t={time_point}d: {ode_err}")
                return np.full(len(df_subset), 1e10) # Return large penalty

        # Ensure prediction doesn't go below y0 or significantly above 1
        y_predicted_all[i] = np.maximum(y0, np.minimum(y_pred, 1.0 + eps))

    # Final check for NaNs or Infs in predictions, which cause curve_fit to fail
    if not np.all(np.isfinite(y_predicted_all)):
        warnings.warn("Non-finite values detected in predictions. Check model/parameters.")
        return np.full(len(df_subset), 1e10) # Return large penalty

    return y_predicted_all

# --- 4. Perform Global Fit (Multi-Batch) ---
print("\n--- Performing Global Fit (Multi-Batch Šesták–Berggren Model) ---")
print(f"Fitting global Ea, n, m, and batch-specific log(A) for {len(unique_batches)} batches.")
if apply_weighting:
    print(f"Applying weighting: 5°C data points have {weight_factor_5C}x more weight.")
else:
    print("Weighting is disabled.")


# Prepare data for curve_fit
x_data_fit = df.index
y_data_fit = df['y'].values # Target variable

# --- Define Weights (sigma) ---
# Weights are only passed to curve_fit if apply_weighting is True
sigma_weights = None
if apply_weighting:
    sigma_weights = np.ones_like(y_data_fit) # Default sigma = 1
    # Assign smaller sigma (higher weight) to 5C data points
    sigma_weights[df['Temperature_C'] == 5] = 1.0 / weight_factor_5C

# --- Initial Guesses (p0) ---
# May need further adjustment based on HMWP_MAX and weighting
initial_Ea_guess = 80
initial_n_guess = 0.5
initial_m_guess = 1.0
initial_logA_guess = 20

p0 = [initial_Ea_guess, initial_n_guess, initial_m_guess] + [initial_logA_guess] * len(unique_batches)
print(f"Using Initial Guesses (p0): Ea={p0[0]:.1f}, n={p0[1]:.2f}, m={p0[2]:.2f}, logA_batch={p0[3]:.2f} (example)")

# Bounds for parameters
bounds_low = [10, 0, 0] + [-np.inf] * len(unique_batches) # Ea>10, n>=0, m>=0
bounds_high = [300, 5, 5] + [np.inf] * len(unique_batches) # Upper bounds for n, m
bounds = (bounds_low, bounds_high)

fit_successful = False # Initialize flag
try:
    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", category=RuntimeWarning)
        popt, pcov = curve_fit(
            global_model_simulate_multibatch,
            xdata=x_data_fit, # Pass indices
            ydata=y_data_fit,
            p0=p0,
            sigma=sigma_weights, # Pass weights (will be None if apply_weighting is False)
            absolute_sigma=False, # Treat sigma as relative weights
            bounds=bounds,
            method='trf',
            max_nfev=5000,
            ftol=1e-6, xtol=1e-6, gtol=1e-6
        )

    if not np.all(np.isfinite(pcov)):
         warnings.warn("Covariance matrix contains non-finite values. Fit uncertainty is high.")
         perr = np.full(len(popt), np.nan)
    else:
         perr = np.sqrt(np.diag(pcov))

    Ea_fit = popt[0]
    n_fit = popt[1]
    m_fit = popt[2]
    logAs_fit = popt[3:]
    logA_dict_fit = {batch_name: logA for batch_name, logA in zip(unique_batches, logAs_fit)}

    Ea_err = perr[0]
    n_err = perr[1]
    m_err = perr[2]
    logAs_err = perr[3:]
    logA_err_dict = {batch_name: err for batch_name, err in zip(unique_batches, logAs_err)}

    print("\n--- Fit Results ---")
    print(f"Global Ea = {Ea_fit:.2f} +/- {Ea_err:.2f} kJ/mol")
    print(f"Global n = {n_fit:.3f} +/- {n_err:.3f}")
    print(f"Global m = {m_fit:.3f} +/- {m_err:.3f}")
    print("Batch-specific log(A):")
    for batch_name in unique_batches:
        logA_val = logA_dict_fit[batch_name]
        logA_err_val = logA_err_dict[batch_name]
        A_val = np.exp(logA_val)
        print(f"  '{batch_name}': {logA_val:.3f} +/- {logA_err_val:.3f} (A ~ {A_val:.2e})")

    fit_successful = True

    # --- 5. Calculate Goodness-of-Fit Metrics ---
    print("\n--- Goodness-of-Fit ---")
    y_predicted_final = global_model_simulate_multibatch(df.index, Ea_fit, n_fit, m_fit, *logAs_fit)

    if np.any(y_predicted_final > 1e9):
         print("  Error: Final prediction contains penalty values. Fit metrics are unreliable.")
         r_squared = np.nan
         rmse = np.nan
    else:
        # Calculate metrics based on unweighted residuals for overall assessment
        hmwp_predicted_final = y_predicted_final * HMWP_MAX
        hmwp_actual = df['HMWP'].values
        residuals = hmwp_actual - hmwp_predicted_final
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hmwp_actual - np.mean(hmwp_actual))**2)
        if ss_tot == 0:
            r_squared = 1.0 if np.isclose(ss_res, 0) else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        mse = np.mean(residuals**2)
        rmse = math.sqrt(mse)
        print(f"  Overall R-squared (R²) = {r_squared:.4f}")
        print(f"  Overall RMSE = {rmse:.4f} (% HMWP)")

except RuntimeError as e:
    print(f"\nERROR: curve_fit failed to converge: {e}")
    print("=> ACTION: Try adjusting HMWP_MAX (line 13) and/or Initial Guesses (p0, lines 152-156).")
    fit_successful = False
except Exception as e:
     print(f"\nERROR during fitting: {e}")
     fit_successful = False

# --- 6. Prediction and Plotting ---
# (Plotting section remains the same as before)
print("\n--- Generating Plot ---")
n_batches = len(unique_batches)
n_cols = 2
n_rows = math.ceil(n_batches / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
axes_flat = axes.flatten()

temp_colors = {5: 'blue', 25: 'green', 30: 'orange', 37: 'red', 40: 'purple'}
temp_markers = {5: 'o', 25: 's', 30: '^', 37: 'D', 40: '*'}
plot_handles = []
plot_labels = []

for i, batch_name in enumerate(unique_batches):
    if i >= len(axes_flat):
        print(f"Warning: More batches ({n_batches}) than available subplots ({len(axes_flat)}). Skipping plot for batch '{batch_name}'. Adjust n_cols/n_rows.")
        break
    ax = axes_flat[i]
    batch_data = df[df['Experiment'] == batch_name].sort_values(by=['Temperature_C', 'Time_days'])

    # Plot experimental data
    for temp_C in sorted(batch_data['Temperature_C'].unique()):
        temp_batch_data = batch_data[batch_data['Temperature_C'] == temp_C]
        color = temp_colors.get(temp_C, 'black')
        marker = temp_markers.get(temp_C, 'x')
        handle = ax.scatter(temp_batch_data['Time_days'], temp_batch_data['HMWP'],
                            color=color, marker=marker, label=f'{temp_C}°C Data')
        if f'{temp_C}°C Data' not in plot_labels:
            plot_labels.append(f'{temp_C}°C Data')
            plot_handles.append(handle)

    # Plot simulated curves if fit was successful
    if fit_successful:
        logA_val = logA_dict_fit[batch_name]
        A_val = np.exp(logA_val)
        y0_val = y0_per_batch[batch_name]
        plot_time_points = np.linspace(0, PREDICTION_YEARS * 365, 200)

        for temp_C in sorted(batch_data['Temperature_C'].unique()):
            temp_K = temp_C + 273.15
            color = temp_colors.get(temp_C, 'black')
            k_sim = A_val * np.exp(-Ea_fit / (R * temp_K))

            try:
                sol_plot = solve_ivp(
                    model_ode, t_span=[0, plot_time_points[-1]], y0=[y0_val],
                    t_eval=plot_time_points, args=(k_sim, n_fit, m_fit),
                    method='RK45', rtol=1e-6, atol=1e-8
                )
                if sol_plot.status == 0:
                    hmwp_sim = sol_plot.y[0] * HMWP_MAX
                    line_handle, = ax.plot(sol_plot.t, hmwp_sim, '-', color=color, label=f'{temp_C}°C Model')
                    if f'{temp_C}°C Model' not in plot_labels:
                         plot_labels.append(f'{temp_C}°C Model')
                         plot_handles.append(line_handle)
                else:
                    print(f"Warning: ODE solver failed during plotting simulation for {batch_name} {temp_C}°C.")
            except Exception as plot_ode_err:
                 print(f"Warning: ODE solver error during plotting simulation for {batch_name} {temp_C}°C: {plot_ode_err}")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("HMWP (%)")
    ax.set_title(f"Batch: {batch_name}")
    ax.grid(True)
    ax.set_ylim(bottom=0)
    max_hmwp_batch = batch_data['HMWP'].max() if not batch_data.empty else 0
    ax.set_ylim(top=max_hmwp_batch * 1.5 if max_hmwp_batch > 0 else 1)

# Hide any unused subplots
for j in range(i + 1, len(axes_flat)):
    fig.delaxes(axes_flat[j])

# Create a single legend
if plot_handles:
    try:
        # Sort labels primarily by temperature, then by type (Data/Model)
        def sort_key(label):
            parts = label.split('°C')
            temp = int(parts[0]) if parts[0].isdigit() else 999
            type_sort = 0 if 'Data' in label else 1
            return (temp, type_sort)

        label_order = sorted(plot_labels, key=sort_key)
        handle_dict = dict(zip(plot_labels, plot_handles))
        sorted_handles = [handle_dict[lbl] for lbl in label_order]
        fig.legend(sorted_handles, label_order, bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize='medium')
    except Exception as e:
        print(f"Could not sort legend: {e}")
        fig.legend(plot_handles, plot_labels, bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize='medium')

fig.suptitle(f"HMWP Formation: Multi-Batch Global Fit (Using HMWP_MAX = {HMWP_MAX}, Weighted={apply_weighting})", fontsize=14)
fig.tight_layout(rect=[0, 0, 0.85, 0.96])
plt.show()

print("\n--- Script Finished ---")
if not fit_successful:
    print("--- >>> Fitting Failed <<< ---")
    print("Recommendation: Adjust HMWP_MAX (line 13) and/or Initial Guesses (p0, lines 152-156) and rerun.")