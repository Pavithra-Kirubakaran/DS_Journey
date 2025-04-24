import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import linregress
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
# Import metrics from sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress RuntimeWarning from curve_fit/integration and OptimizeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
# warnings.filterwarnings("ignore", category=np.RankWarning) # Already commented out

# =============================================================================
# Configuration
# =============================================================================
excel_filename = 'Data/stability_data.xlsx'
predictions_filename = 'purity_predictions_autocat_3yr.xlsx'
max_training_days = 90
prediction_days = 3 * 365 # 3 years
# Initial guesses and bounds for fitting K, n, m
# [K_guess, n_guess, m_guess]
initial_params_guess = [1e-3, 0.5, 1.0]
# [(K_min, n_min, m_min), (K_max, n_max, m_max)]
param_bounds = ([0, 0, 0], [np.inf, 5, 5]) # K,n,m >=0, n,m <=5 arbitrary upper limit

# =============================================================================
# 1. Data Loading and Preparation
# =============================================================================
try:
    df_all = pd.read_excel(excel_filename)
    print(f"--- Successfully loaded data from '{excel_filename}' ---")
    print(f"Found {len(df_all)} total data points.")
except FileNotFoundError:
    print(f"ERROR: Excel file '{excel_filename}' not found.")
    print("Please run the 'create_excel_data' script first to generate the data file.")
    exit() # Exit if the data file doesn't exist
except Exception as e:
    print(f"ERROR: Could not read Excel file '{excel_filename}': {e}")
    exit()

# Basic Data Cleaning & Preparation
df_all = df_all.dropna(subset=['Experiment', 'Temperature_C', 'Time_days', 'Purity'])
df_all = df_all.astype({'Temperature_C': float, 'Time_days': float, 'Purity': float})
df_all = df_all[df_all['Purity'] >= 0] # Ensure purity is non-negative

# Convert Temperature to Kelvin
df_all['Temperature_K'] = df_all['Temperature_C'] + 273.15

# Calculate Purity_t0 and Fraction_Remaining
df_all['Time_rank'] = df_all.groupby(['Experiment', 'Temperature_C'])['Time_days'].rank(method='first')
purity_t0_map = df_all.loc[df_all['Time_rank'] == 1].set_index(['Experiment', 'Temperature_C'])['Purity']
df_all['Purity_t0'] = df_all.set_index(['Experiment', 'Temperature_C']).index.map(purity_t0_map)
df_all['Purity_t0'] = df_all['Purity_t0'].fillna(method='ffill').fillna(method='bfill')
df_all['Purity_t0'] = df_all['Purity_t0'].replace(0, np.nan)

df_all['Fraction_Remaining'] = np.nan
valid_p0_idx = df_all['Purity_t0'].notna() & (df_all['Purity_t0'] > 0)
df_all.loc[valid_p0_idx, 'Fraction_Remaining'] = (df_all.loc[valid_p0_idx, 'Purity'] / df_all.loc[valid_p0_idx, 'Purity_t0'])
df_all['Fraction_Remaining'] = df_all['Fraction_Remaining'].fillna(1.0)
df_all['Fraction_Remaining'] = df_all['Fraction_Remaining'].clip(upper=1.0)

# Calculate Conversion y (alpha) = 1 - Fraction_Remaining
df_all['Conversion_y'] = 1.0 - df_all['Fraction_Remaining']
# Ensure conversion is between 0 and 1 (or slightly less than 1 for numerical stability)
df_all['Conversion_y'] = df_all['Conversion_y'].clip(lower=0.0, upper=0.999999)

# =============================================================================
# 2. Filter Data for Training
# =============================================================================
df_train = df_all[
    (df_all['Time_days'] > 0) &
    (df_all['Time_days'] <= max_training_days) &
    df_all['Conversion_y'].notna()
].copy()

print(f"\n--- Data Filtering for Training ---")
print(f"Using data points with 0 < Time_days <= {max_training_days} days for training.")
print(f"Number of training data points: {len(df_train)}")
if len(df_train) == 0:
    print("ERROR: No data available for training based on the filter criteria.")
    exit()
print("-" * 40)

# =============================================================================
# 3. Define Kinetic Model (Differential Equation) and Solver Function
# =============================================================================

def reaction_ode(t, y, K, n, m):
    """
    Differential equation for dy/dt = K * y^n * (1-y)^m.
    Handles potential numerical issues at y=0 and y=1.
    y is treated as a scalar here.
    """
    # Add small epsilon to avoid log(0) or 0^0 issues if n or m are < 1
    epsilon = 1e-12
    y_safe = np.clip(y, epsilon, 1.0 - epsilon)
    term1 = y_safe**n
    term2 = (1.0 - y_safe)**m
    dydt = K * term1 * term2
    # Ensure rate is non-negative
    return max(0.0, dydt)

def solve_reaction_model(t_eval, K, n, m):
    """
    Solves the ODE model for given parameters and returns conversion y at t_eval points.
    """
    y0 = [0.0] # Initial condition: conversion y = 0 at t = 0
    t_span = [0, max(t_eval) * 1.01] # Ensure span covers all evaluation points

    try:
        sol = solve_ivp(
            reaction_ode,
            t_span,
            y0,
            args=(K, n, m),
            t_eval=t_eval,
            method='RK45', # Common robust solver
            dense_output=True # Allows interpolation if needed
        )
        if sol.status != 0:
            # print(f"Warning: ODE solver failed with status {sol.status} for K={K:.2e}, n={n:.2f}, m={m:.2f}")
            # Return NaN or previous value if solver fails? For curve_fit, better to let it fail/return large error.
             return np.full_like(t_eval, np.nan) # Return NaNs if solver failed

        # Extract solution at t_eval points
        y_solution = sol.y[0]
        # Ensure solution stays within bounds [0, 1]
        return np.clip(y_solution, 0.0, 1.0)

    except Exception as e:
        # print(f"Exception during ODE solve: {e} for K={K:.2e}, n={n:.2f}, m={m:.2f}")
        return np.full_like(t_eval, np.nan) # Return NaNs on exception

# Wrapper function for curve_fit - takes time points and returns conversion y
def model_for_curvefit(t_data, K, n, m):
    """Wrapper for solve_reaction_model to be used with curve_fit."""
    # Need to sort t_data for solve_ivp's t_eval argument
    sort_indices = np.argsort(t_data)
    t_sorted = t_data[sort_indices]

    y_solved_sorted = solve_reaction_model(t_sorted, K, n, m)

    # Need to return y_solved in the original order of t_data
    # Create an array to hold the result in the original order
    y_solved_original_order = np.full_like(t_data, np.nan)
    # Place the sorted results back into the original order
    y_solved_original_order[sort_indices] = y_solved_sorted

    # If solver failed, return array of NaNs or large numbers to guide curve_fit away
    if np.isnan(y_solved_original_order).any():
         return np.full_like(t_data, 1e10) # Return large number if solve failed

    return y_solved_original_order


# Gas constant (J/mol/K)
R = 8.314

# =============================================================================
# 4. Fit Kinetics for Each Condition (Experiment + Temperature)
# =============================================================================
print("--- Fitting Autocatalytic Kinetics (Using Training Data) ---")
# Dictionary stores tuple: {(Experiment, temp_K): (K, n, m)}
fitted_params = {}
failed_fits = []

# Group training data by Experiment and Temperature
grouped_train = df_train.groupby(['Experiment', 'Temperature_K'])

for name, group in grouped_train:
    experiment, temp_k = name
    temp_c = temp_k - 273.15
    print(f"Fitting for: {experiment} at {temp_c:.1f}°C ({temp_k:.2f} K)")

    # Use unique time points for fitting, average y if duplicates exist
    group_agg = group.groupby('Time_days')['Conversion_y'].mean().reset_index()
    t_data = group_agg['Time_days'].values
    y_data = group_agg['Conversion_y'].values # Target is conversion y

    # Ensure data is suitable for fitting
    if len(t_data) < 3: # Need more points for 3 parameters
        print(f"  Skipping: Insufficient unique data points ({len(t_data)}) for fitting K, n, m.")
        failed_fits.append(name)
        continue
    if np.all(y_data <= 1e-6): # Check if any significant conversion occurred
        print(f"  Skipping: No significant conversion observed.")
        failed_fits.append(name)
        continue

    try:
        # Perform the non-linear curve fit using the ODE solver wrapper
        params, covariance = curve_fit(
            model_for_curvefit,
            t_data,
            y_data,
            p0=initial_params_guess,
            bounds=param_bounds,
            method='trf', # Trust Region Reflective often better with bounds
            max_nfev=1000 # Max function evaluations (calls to model_for_curvefit)
        )
        K_fit, n_fit, m_fit = params

        # Check for reasonable fit parameters and uncertainty
        perr = np.sqrt(np.diag(covariance))
        if np.inf in perr or np.nan in perr:
             print(f"  Warning: Fit uncertainty calculation failed. Check fit quality.")
             # Optionally treat as failed fit
             # failed_fits.append(name)
             # continue

        fitted_params[name] = (K_fit, n_fit, m_fit)
        print(f"  Found K={K_fit:.4e}, n={n_fit:.3f}, m={m_fit:.3f}")

    except (RuntimeError, ValueError, Exception) as e:
        print(f"  ERROR: Could not converge fit for {experiment} at {temp_c:.1f}°C. Skipping. ({type(e).__name__})")
        failed_fits.append(name)

print("-" * 40)
print("Fitted parameters (K, n, m) determined for:")
if fitted_params:
    for (exp, temp_k), (k, n, m) in sorted(fitted_params.items()):
         print(f"  {exp} @ {temp_k-273.15:.1f}°C : K={k:.4e}, n={n:.3f}, m={m:.3f}")
else:
    print("  No successful fits were obtained.")
if failed_fits:
    print("\nFailed fits (or skipped) for:")
    for exp, temp_k in sorted(failed_fits):
        print(f"  {exp} @ {temp_k-273.15:.1f}°C")
print("-" * 40)


# =============================================================================
# 5. Arrhenius Plot and Calculation (Using fitted K) & Average n, m
# =============================================================================
print("--- Arrhenius Analysis & Averaging n, m ---")
Ea = None
A = None
n_avg = None
m_avg = None
arrhenius_data_ok = False

successful_fits = list(fitted_params.keys())
if len(successful_fits) >= 2:
    # Extract K, n, m from successful fits
    k_values = [params[0] for params in fitted_params.values()]
    n_values = [params[1] for params in fitted_params.values()]
    m_values = [params[2] for params in fitted_params.values()]
    temps_K_arrh = np.array([temp_k for (_, temp_k) in successful_fits])

    # Average n and m
    n_avg = np.mean(n_values)
    m_avg = np.mean(m_values)
    n_std = np.std(n_values)
    m_std = np.std(m_values)
    print(f"Average fitted reaction orders:")
    print(f"  Average n = {n_avg:.3f} (StdDev: {n_std:.3f})")
    print(f"  Average m = {m_avg:.3f} (StdDev: {m_std:.3f})")
    print("(Using these average n, m for predictions)")

    # Prepare data for Arrhenius plot (using K values)
    valid_k_indices = np.array(k_values) > 1e-15
    if np.sum(valid_k_indices) >= 2:
        temps_K_arrh = temps_K_arrh[valid_k_indices]
        k_values_arrh = np.array(k_values)[valid_k_indices]

        # Check if we have at least two unique temperatures with valid K
        unique_temps_for_arrh = np.unique(temps_K_arrh)
        if len(unique_temps_for_arrh) >= 2:
            # Average K if multiple experiments exist at the same temperature
            arrh_temps_final = []
            arrh_ln_k_avg = []
            temp_k_map = {}
            for temp, k_val in zip(temps_K_arrh, k_values_arrh):
                if temp not in temp_k_map: temp_k_map[temp] = []
                temp_k_map[temp].append(k_val)

            for temp, k_list in temp_k_map.items():
                arrh_temps_final.append(temp)
                arrh_ln_k_avg.append(np.log(np.mean(k_list)))

            if len(arrh_temps_final) >= 2:
                inv_T = 1.0 / np.array(arrh_temps_final)
                ln_k = np.array(arrh_ln_k_avg)

                try:
                    # Perform linear regression: ln(k) = ln(A) - Ea / (R * T)
                    slope, intercept, r_value, p_value, std_err = linregress(inv_T, ln_k)

                    Ea = -slope * R  # Activation Energy in J/mol
                    A = np.exp(intercept) # Pre-exponential factor in day^-1

                    if Ea > 0 and A > 0:
                        print(f"\nArrhenius Fit Results (using avg K per temp):")
                        print(f"  Activation Energy (Ea): {Ea / 1000:.2f} kJ/mol")
                        print(f"  Pre-exponential Factor (A): {A:.4e} day^-1")
                        print(f"  R-squared (Arrhenius fit): {r_value**2:.4f}")
                        arrhenius_data_ok = True

                        # --- Plot Arrhenius Plot ---
                        # (Plotting code similar to previous version, using K values)
                        plt.figure(figsize=(7, 5))
                        # Plot individual K values
                        plt.scatter(1.0/temps_K_arrh, np.log(k_values_arrh), marker='x', s=40, alpha=0.6, label='ln(K) from fits')
                        # Plot average ln(K) used for fit
                        plt.scatter(inv_T, ln_k, marker='o', s=80, facecolors='none', edgecolors='blue', label='Avg ln(K) per Temp')
                        inv_T_fit = np.linspace(min(inv_T), max(inv_T), 50)
                        ln_k_fit = intercept + slope * inv_T_fit
                        plt.plot(inv_T_fit, ln_k_fit, 'r-', label=f'Arrhenius Fit (Ea={Ea/1000:.1f} kJ/mol)')
                        plt.xlabel("1 / Temperature (1/K)")
                        plt.ylabel("ln(K)  (K in day^-1)")
                        plt.title("Arrhenius Plot (Autocatalytic Model)")
                        plt.legend()
                        plt.grid(True, linestyle=':', alpha=0.7)
                        plt.tight_layout()
                        plt.show()
                        # --- End Arrhenius Plot ---
                    else:
                        print("Arrhenius fit resulted in non-physical parameters (e.g., Ea <= 0).")
                except ValueError as e:
                     print(f"Error during linear regression for Arrhenius plot: {e}")
            else:
                print("Need valid rate constants from at least two different temperatures for Arrhenius plot.")
        else:
            print("Need valid rate constants from at least two unique temperatures for Arrhenius plot.")
    else:
        print("Need at least two valid positive rate constants for Arrhenius plot.")
else:
    print("Need successful fits from at least two conditions for Arrhenius plot.")

if not arrhenius_data_ok:
     print("\nSkipping prediction and evaluation steps as valid Arrhenius parameters were not determined.")
     # Optionally, allow prediction using individual K,n,m if Arrhenius failed? No, stick to Arrhenius for extrapolation.
     exit()

print("-" * 40)


# =============================================================================
# 6. Generate Predictions for ALL Experimental Time Points for Evaluation
# =============================================================================
print("--- Generating Predictions for Evaluation using Arrhenius K and Avg n, m ---")
df_eval = df_all[df_all['Purity_t0'].notna() & (df_all['Time_days'] >= 0)].copy()
df_eval['Predicted_Purity'] = np.nan
df_eval['Predicted_Conversion_y'] = np.nan

# Use average n, m determined from fits
pred_n = n_avg
pred_m = m_avg

unique_eval_conditions = df_eval[['Experiment', 'Temperature_K']].drop_duplicates()

for index, row in unique_eval_conditions.iterrows():
    experiment = row['Experiment']
    temp_k = row['Temperature_K']

    # Get all time points for this condition
    condition_idx = (df_eval['Experiment'] == experiment) & (df_eval['Temperature_K'] == temp_k)
    t_points_eval = df_eval.loc[condition_idx, 'Time_days'].unique()
    t_points_eval = np.sort(t_points_eval[t_points_eval >= 0]) # Ensure non-negative sorted times

    if len(t_points_eval) == 0: continue

    # Calculate predicted K using Arrhenius parameters
    K_pred = A * np.exp(-Ea / (R * temp_k))

    # Solve ODE for this condition over its time points
    y_pred_solved = solve_reaction_model(t_points_eval, K_pred, pred_n, pred_m)

    if np.isnan(y_pred_solved).any():
        print(f"Warning: ODE solve failed during evaluation prediction for {experiment} @ {temp_k-273.15:.1f}C.")
        continue

    # Map predictions back to the dataframe
    time_to_y_map = dict(zip(t_points_eval, y_pred_solved))
    df_eval.loc[condition_idx, 'Predicted_Conversion_y'] = df_eval.loc[condition_idx, 'Time_days'].map(time_to_y_map)

# Calculate predicted purity
df_eval['Predicted_Purity'] = (1.0 - df_eval['Predicted_Conversion_y']) * df_eval['Purity_t0']

# Drop rows where prediction failed or data was invalid
df_eval = df_eval.dropna(subset=['Predicted_Purity', 'Purity'])

print(f"Generated predictions for {len(df_eval)} experimental points.")
print("-" * 40)


# =============================================================================
# 7. Calculate Evaluation Metrics (Based on Purity)
# =============================================================================
# (Evaluation metrics calculation code is identical to previous version,
#  using df_eval['Purity'] as y_true and df_eval['Predicted_Purity'] as y_pred)
print("--- Model Evaluation Metrics (Based on Purity Predictions) ---")

# Define function for MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    if np.sum(non_zero_idx) == 0: return np.nan
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

# Separate data for evaluation
eval_train = df_eval[df_eval['Time_days'] <= max_training_days]
eval_valid = df_eval[df_eval['Time_days'] > max_training_days]

metrics_results = {}
for label, df_subset in [('Training', eval_train), ('Validation', eval_valid), ('Overall', df_eval)]:
    if len(df_subset) > 0:
        y_true = df_subset['Purity']
        y_pred = df_subset['Predicted_Purity']
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        # R2 score can be misleading if model is poor, check range
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        metrics_results[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'Count': len(df_subset)}
        print(f"\nMetrics for {label} Data (N={len(df_subset)}):")
        print(f"  Mean Absolute Error (MAE):   {mae:.3f} % Purity")
        print(f"  Root Mean Squared Error (RMSE):{rmse:.3f} % Purity")
        print(f"  R-squared (R2):              {r2:.4f}")
        print(f"  Mean Abs Percentage Error (MAPE):{mape:.2f} %")
    else:
        print(f"\nNo data available for {label} set evaluation.")
print("-" * 40)

# =============================================================================
# 8. Residual Plot (Based on Purity)
# =============================================================================
# (Residual plot code is identical to previous version, using Purity)
print("--- Generating Residual Plot (Based on Purity) ---")
df_eval['Residual'] = df_eval['Purity'] - df_eval['Predicted_Purity']
plt.figure(figsize=(10, 6))
plt.scatter(df_eval['Predicted_Purity'], df_eval['Residual'], alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Purity (%)")
plt.ylabel("Residual (Actual - Predicted Purity %)")
plt.title("Residual Plot (Autocatalytic Model)")
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()