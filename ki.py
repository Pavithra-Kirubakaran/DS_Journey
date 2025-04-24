import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import sys
import math # For log in AIC/BIC

# --- Constants ---
R = 8.314  # J/(mol*K)

# --- !!! USER INPUTS !!! ---
excel_file_path = "/content/stability_data.xlsx" # <<< CHANGE THIS to your data file

# List of attribute columns from the Excel file to model
ATTRIBUTES_TO_MODEL = [
    'HMWP (%)',
    'Monomer (%)',
    'Acidic Variants (%)',
    'Basic Variants (%)',
    'Main Peak Area (%)',
    'Total Fragments Area (%)',
    'LC + HC Fragments (%)',
    'Monomer Area (%)'
]

# Define the time column name and units/limits
TIME_COLUMN = 'Time Point (Months)' # <<< Ensure this matches your Excel header
TRAIN_UP_TO_TIME_UNIT = 12 # Use data up to this time unit (e.g., 12 months) for training
EXTRAPOLATE_TO_TIME_UNIT = 36 # Extrapolate predictions up to this time unit (e.g., 36 months)

# Base column mapping (Time, Temperature, Batch)
column_mapping = {
    'Temperature (°C)': 'T_c',
    TIME_COLUMN: 'time', # Internal name for the time column (unit defined above)
    'Batch': 'batch'
    # Specific attributes (like HMWP %) are handled in the loop
}

# Optional: Define specific settings per attribute to override defaults/heuristics
# Use 'initial_sign': 1 for known formation, -1 for known degradation
# Use 'y_scale' for attributes not on a 0-100 scale or with a different known max
attribute_settings = {
    'HMWP (%)': {'initial_sign': 1, 'y_scale': 10.0}, # Example: HMWP forms, max expected ~10%
    'Monomer (%)': {'initial_sign': -1, 'y_scale': 100.0}, # Example: Monomer degrades from 100%
    # 'Acidic Variants (%)': {'initial_sign': 1}, # Example if known formation
    # Add other attributes here if you need to override the heuristic sign or default y_scale=100
}
# ---

# --- Read Data From Excel ---
try:
    print(f"--- Reading Data From Excel: {excel_file_path} ---")
    data_df_full = pd.read_excel(excel_file_path)
    print(f"Successfully read {len(data_df_full)} data points.")
except FileNotFoundError:
    print(f"ERROR: Excel file not found at '{excel_file_path}'.")
    sys.exit(f"Exiting: File '{excel_file_path}' not found.")
except Exception as e:
    print(f"ERROR: Could not read Excel file. Error: {e}")
    sys.exit(f"Exiting: Error reading file.")

# --- Initial Data Preparation (Common Columns) ---
print("--- Preparing Data ---")
required_base_cols = list(column_mapping.keys())
missing_base_cols = [col for col in required_base_cols if col not in data_df_full.columns]
if missing_base_cols:
    print(f"ERROR: Missing required base columns: {missing_base_cols}.")
    sys.exit("Exiting: Missing base columns.")

# Check if all attributes to model exist
missing_attributes = [attr for attr in ATTRIBUTES_TO_MODEL if attr not in data_df_full.columns]
if missing_attributes:
    print(f"ERROR: Missing attribute columns to model: {missing_attributes}.")
    sys.exit("Exiting: Missing attribute columns.")

data_df_full = data_df_full.rename(columns=column_mapping)
data_df_full['T_k'] = data_df_full['T_c'] + 273.15

# Ensure base columns are numeric where expected
base_numeric_cols = ['T_c', 'time', 'T_k']
for col in base_numeric_cols:
     data_df_full[col] = pd.to_numeric(data_df_full[col], errors='coerce')
initial_rows = len(data_df_full)
data_df_full.dropna(subset=base_numeric_cols, inplace=True)
if len(data_df_full) < initial_rows:
     print(f"WARNING: Dropped {initial_rows - len(data_df_full)} rows due to non-numeric base data (Time, Temp).")
if data_df_full.empty:
     sys.exit("Exiting: No valid numeric base data.")

# --- Define Unified UDE Model Class ---
class UDEModel(nn.Module):
    """
    Unified Differential Equation (UDE) model combining Arrhenius kinetics
    with a neural network correction term and flexible reaction orders.
    Models dy/dt = sign * k * (y/y_scale)^n * (1 - y/y_scale)^m * y_scale
    where k = k_arrhenius * k_nn
    """
    def __init__(self, initial_logA=np.log(1e-1), initial_logEa=np.log(55000),
                 initial_log_n=np.log(1.0), initial_log_m=np.log(1.0),
                 initial_sign=-1.0, y_scale=100.0):
        """
        Initializes the UDEModel. See previous versions for detailed Args.
        """
        super(UDEModel, self).__init__()
        self.logA = nn.Parameter(torch.tensor(initial_logA, dtype=torch.float32))
        self.logEa = nn.Parameter(torch.tensor(initial_logEa, dtype=torch.float32))
        self.log_n = nn.Parameter(torch.tensor(initial_log_n, dtype=torch.float32))
        self.log_m = nn.Parameter(torch.tensor(initial_log_m, dtype=torch.float32))
        self.sign_param = nn.Parameter(torch.tensor(initial_sign, dtype=torch.float32))
        self.y_scale = y_scale

        # --- CHANGE 1: Slightly Increased NN Complexity ---
        # Increased hidden layer size back to 20 for more flexibility
        hidden_size = 20 # Changed from 10 back to 20
        self.nn_k = nn.Sequential(
            nn.Linear(1, hidden_size), # Input layer (1 feature: T_k)
            nn.Tanh(),             # Activation function
            nn.Linear(hidden_size, 1) # Output layer (1 feature: log(k_nn))
        )
        # Initialize NN weights
        for layer in self.nn_k:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, y, t, T_k):
        """ Calculates dy/dt based on the UDE model. """
        A = torch.exp(self.logA)
        Ea = torch.exp(self.logEa)
        Ea_clamped = torch.clamp(Ea, min=1000)
        k_arrhenius = A * torch.exp(-Ea_clamped / (R * T_k))
        if T_k.ndim == 0: T_k_input = T_k.unsqueeze(0)
        elif T_k.ndim == 1: T_k_input = T_k
        else: T_k_input = T_k.view(-1, 1)
        log_k_nn = self.nn_k(T_k_input)
        k_nn = torch.exp(log_k_nn).squeeze()
        k = k_arrhenius * k_nn
        n = torch.exp(self.log_n)
        m = torch.exp(self.log_m)
        y_normalized = torch.clamp(y / self.y_scale, min=1e-9, max=1.0 - 1e-9)
        one_minus_y_norm = torch.clamp(1.0 - y_normalized, min=1e-9)
        signed_k = k * torch.tanh(self.sign_param)
        dydt = signed_k * torch.pow(y_normalized, n) * torch.pow(one_minus_y_norm, m) * self.y_scale
        return dydt

    def get_params_dict(self):
        """ Returns a dictionary of the interpretable parameters. """
        with torch.no_grad():
            params = {
                'A': torch.exp(self.logA).item(), 'Ea': torch.exp(self.logEa).item(),
                'n': torch.exp(self.log_n).item(), 'm': torch.exp(self.log_m).item(),
                'sign': torch.tanh(self.sign_param).item(), 'y_scale': self.y_scale
            }
        return params

    def get_num_params(self):
        """ Calculates the total number of trainable parameters including NN. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --- Differentiable ODE Solver (RK4) ---
def solve_ode_rk4_torch(ode_func, y0, t_eval):
    """ Solves ODE dy/dt = ode_func(y, t) using RK4 with PyTorch. """
    y_i = y0.clone()
    # Ensure t_eval is sorted for the solver steps
    t_eval_sorted, sort_indices = torch.sort(t_eval)
    # Calculate predictions based on sorted times
    dts = t_eval_sorted[1:] - t_eval_sorted[:-1]
    y_pred_dict = {t_eval_sorted[0].item(): y_i.clone()}
    for i in range(len(t_eval_sorted) - 1):
        t_i = t_eval_sorted[i]; dt = dts[i]
        if dt <= 1e-9:
            y_pred_dict[t_eval_sorted[i+1].item()] = y_i.clone(); continue
        k1 = ode_func(y_i, t_i)
        k2 = ode_func(y_i + 0.5 * dt * k1, t_i + 0.5 * dt)
        k3 = ode_func(y_i + 0.5 * dt * k2, t_i + 0.5 * dt)
        k4 = ode_func(y_i + dt * k3, t_i + dt)
        y_i = y_i + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        y_i = torch.clamp(y_i, min=0.0) # Ensure non-negativity
        y_pred_dict[t_eval_sorted[i+1].item()] = y_i.clone()

    # Reconstruct the final prediction tensor in the *original* order of t_eval
    try:
        y_pred_final = [y_pred_dict[t.item()] for t in t_eval]
    except KeyError as e:
        print(f"KeyError during prediction reconstruction: {e}. Trying sorted.")
        # Fallback: Try to reconstruct based on sorted times if lengths match
        if len(y_pred_dict) == len(t_eval):
            y_pred_final = [y_pred_dict[t.item()] for t in t_eval_sorted]
        else:
            print(f"Error: Prediction dictionary size ({len(y_pred_dict)}) mismatch with t_eval size ({len(t_eval)}).")
            return torch.tensor([]) # Return empty on mismatch

    if len(y_pred_final) != len(t_eval):
        print(f"Warning: Prediction length mismatch ({len(y_pred_final)} vs {len(t_eval)}).")
        return torch.tensor([])
    return torch.stack(y_pred_final)

# --- Training Function ---
def train_ude_parameters(ude_model, data_grouped_by_temp, y0_tensor, model_name, epochs=3000, lr=0.01):
    """ Trains the parameters of the UDEModel with weight decay and adjusted LR schedule. """
    # Using weight_decay for L2 regularization
    optimizer = optim.Adam(ude_model.parameters(), lr=lr, weight_decay=1e-4) # Keep weight decay
    # --- CHANGE 2: Adjusted LR Schedule ---
    # Decrease LR slightly later (step_size changed from epochs//4 to epochs//3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    losses = []
    print(f"--- Starting UDE Training ({model_name}) ---")
    start_time = time.time()
    for epoch in range(epochs):
        ude_model.train(); optimizer.zero_grad()
        total_loss = 0.0; total_points = 0
        for T_k, temp_data in data_grouped_by_temp.items():
            t_all = temp_data['time']; y_target = temp_data['concentration']
            T_k_tensor = torch.tensor(T_k, dtype=torch.float32)
            ode_func_for_temp = lambda y, t: ude_model(y, t, T_k_tensor)
            y_pred = solve_ode_rk4_torch(ode_func_for_temp, y0_tensor.clone(), t_all)
            if y_pred.numel() == 0 or y_pred.squeeze().shape != y_target.shape:
                print(f"Warning ({model_name}): Shape mismatch/empty pred at T={T_k-273.15:.1f}C, epoch {epoch+1}. Skip batch.")
                continue
            loss = torch.sum((y_pred.squeeze() - y_target)**2)
            total_loss = total_loss + loss; total_points += len(y_target)
        if total_points == 0:
            print(f"Warning ({model_name}): No points processed epoch {epoch+1}. Stop train."); losses.append(np.nan); return losses, False
        avg_loss = total_loss / total_points; losses.append(avg_loss.item())
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print(f"ERROR ({model_name}): Loss={avg_loss.item()} epoch {epoch+1}. Stop train."); return losses, False
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(ude_model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            param_str = ", ".join([f"{k}={v:.3g}" for k, v in ude_model.get_params_dict().items() if k != 'y_scale'])
            print(f'{model_name} - Ep [{epoch+1}/{epochs}], Loss: {avg_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.1e}, P: {param_str}')
    end_time = time.time()
    print(f"Training finished for {model_name} in {end_time - start_time:.2f} seconds.")
    fit_successful = not any(np.isnan(p) or np.isinf(p) for p in ude_model.get_params_dict().values() if isinstance(p, float))
    if fit_successful: print(f"Final Params ({model_name}) -> {', '.join([f'{k}={v:.3g}' for k,v in ude_model.get_params_dict().items()])}")
    else: print(f"WARNING ({model_name}): Final parameters contain NaN/Inf.")
    print("-" * 30)
    return losses, fit_successful

# --- Evaluation Metrics Calculation ---
def get_predictions_and_actuals(ude_model, data_df, y0_tensor):
    """ Generates model predictions for the given data. """
    ude_model.eval(); all_preds = []; all_actuals = []
    grouped = data_df.sort_values(by=['T_k', 'time']).groupby('T_k')
    with torch.no_grad():
        for T_k, group in grouped:
            eval_times = group['time'].unique()
            t_eval = torch.tensor(eval_times, dtype=torch.float32)
            y_actual_group = group # Keep full group to map predictions back
            T_k_tensor = torch.tensor(T_k, dtype=torch.float32)
            ode_func_for_temp = lambda y, t: ude_model(y, t, T_k_tensor)
            y_pred_torch = solve_ode_rk4_torch(ode_func_for_temp, y0_tensor.clone(), t_eval)
            if y_pred_torch.numel() == 0 or y_pred_torch.squeeze().shape[0] != len(eval_times):
                print(f"Warning: Skip eval T={T_k-273.15:.1f}C due to pred shape mismatch.")
                continue
            # Map predictions for unique times back to all original time points
            pred_map = dict(zip(eval_times, y_pred_torch.squeeze().cpu().numpy()))
            y_pred_np = y_actual_group['time'].map(pred_map).values
            y_actual_np = y_actual_group['concentration'].values
            # Filter out NaNs that might arise from mapping or original data
            valid_idx = ~np.isnan(y_pred_np) & ~np.isnan(y_actual_np)
            all_preds.extend(y_pred_np[valid_idx])
            all_actuals.extend(y_actual_np[valid_idx])
    return np.array(all_actuals), np.array(all_preds)

def calculate_fit_metrics(y_actual, y_predicted, num_observations, num_params, model_name):
    """ Calculates R2, RMSE, AIC, and BIC metrics. """
    if len(y_actual) < 2 or len(y_predicted) < 2 or len(y_actual) != len(y_predicted):
        print(f"Warning ({model_name}): Mismatch len actual ({len(y_actual)}) vs pred ({len(y_predicted)})."); return {'R2': np.nan, 'RMSE': np.nan, 'AIC': np.nan, 'BIC': np.nan}
    if num_observations <= num_params: print(f"Warning ({model_name}): Obs <= Params ({num_observations}<={num_params}). AIC/BIC invalid.")
    metrics = {'R2': np.nan, 'RMSE': np.nan, 'AIC': np.nan, 'BIC': np.nan}
    try:
        metrics['R2'] = r2_score(y_actual, y_predicted)
        mse = mean_squared_error(y_actual, y_predicted)
        metrics['RMSE'] = np.sqrt(mse)
        rss = mse * num_observations
        if num_observations > num_params and rss > 1e-12:
            log_likelihood_term = num_observations * math.log(rss / num_observations)
            metrics['AIC'] = log_likelihood_term + 2 * num_params
            metrics['BIC'] = log_likelihood_term + num_params * math.log(num_observations)
        elif rss <= 1e-12: print(f"Warning ({model_name}): RSS near zero. AIC/BIC set to -inf."); metrics['AIC'] = -np.inf; metrics['BIC'] = -np.inf
        print(f"Calculated Metrics ({model_name} - Training Data):")
        for key, value in metrics.items(): print(f"  {key}: {value:.4f}")
        return metrics
    except Exception as e: print(f"Error calculating metrics for {model_name}: {e}"); return {'R2': np.nan, 'RMSE': np.nan, 'AIC': np.nan, 'BIC': np.nan}

# --- Plotting Function ---
def plot_results(attribute_name, model, data_df_attribute, y0_tensor,
                 train_limit_time, extrapolate_limit_time, metrics, fit_successful, time_unit_name):
    """ Generates plot for a single attribute with simplified legend and adjusted axes """
    print(f"--- Generating Plot for {attribute_name} ---")
    plt.figure(figsize=(14, 9))
    unique_temps_k_all = np.sort(data_df_attribute['T_k'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps_k_all)))
    temp_color_map = {T_k: colors[i] for i, T_k in enumerate(unique_temps_k_all)}
    batch_markers = ['o', 's', '^', 'D', 'v', '<', '>']
    t_extrapolate_dense = np.linspace(0, extrapolate_limit_time, 200)

    plotted_batch_markers = set()
    # Initialize plot limits with actual data range first
    min_data_y = data_df_attribute['concentration'].min() if not data_df_attribute.empty else 0
    max_data_y = data_df_attribute['concentration'].max() if not data_df_attribute.empty else 1
    min_plot_lim = min_data_y
    max_plot_lim = max_data_y

    # Plot ALL data points
    batches = data_df_attribute['batch'].unique()
    for i, batch_id in enumerate(batches):
        batch_df = data_df_attribute[data_df_attribute['batch'] == batch_id]
        marker = batch_markers[i % len(batch_markers)]
        needs_batch_label = batch_id not in plotted_batch_markers
        batch_label_to_use = f'Batch {batch_id}' if needs_batch_label else None
        for T_k, color in temp_color_map.items():
            temp_batch_df = batch_df[batch_df['T_k'] == T_k]
            if not temp_batch_df.empty:
                is_train_mask = temp_batch_df['time'] <= train_limit_time
                label_for_train = batch_label_to_use if is_train_mask.any() else None
                plt.scatter(temp_batch_df.loc[is_train_mask, 'time'], temp_batch_df.loc[is_train_mask, 'concentration'],
                            edgecolor=color, facecolor=color, marker=marker, s=50, alpha=0.8, zorder=5, label=label_for_train)
                label_for_test = batch_label_to_use if not is_train_mask.any() and label_for_train is None else None
                plt.scatter(temp_batch_df.loc[~is_train_mask, 'time'], temp_batch_df.loc[~is_train_mask, 'concentration'],
                            edgecolor=color, facecolor='none', marker=marker, s=50, alpha=0.8, zorder=5, label=label_for_test)
                if needs_batch_label and (label_for_train is not None or label_for_test is not None):
                    plotted_batch_markers.add(batch_id); needs_batch_label = False; batch_label_to_use = None

    # Plot fit lines and update plot limits based on prediction range
    if fit_successful:
        model.eval()
        with torch.no_grad():
            all_preds_for_lims = [] # Collect all predictions for axis limits
            for j, T_k in enumerate(unique_temps_k_all):
                T_c = T_k - 273.15; color = temp_color_map[T_k]
                T_k_tensor = torch.tensor(T_k, dtype=torch.float32)
                t_dense_torch = torch.tensor(t_extrapolate_dense, dtype=torch.float32)
                ode_func_predict = lambda y, t: model(y, t, T_k_tensor)
                y_pred_torch = solve_ode_rk4_torch(ode_func_predict, y0_tensor.clone(), t_dense_torch)
                if y_pred_torch.numel() > 0:
                    y_pred_np = y_pred_torch.squeeze().cpu().numpy()
                    # Update plot limits based on these predictions
                    min_plot_lim = min(min_plot_lim, np.min(y_pred_np))
                    max_plot_lim = max(max_plot_lim, np.max(y_pred_np))
                    # Plot the line
                    label = f'Fit ({T_c:.0f}°C)'
                    plt.plot(t_extrapolate_dense, y_pred_np, '--', label=label, color=color, linewidth=2)
                else: print(f"Skipping plot line {attribute_name} T={T_c:.0f}C: solver fail.")
    else: plt.text(0.5, 0.5, f'{attribute_name}\nFit Failed', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14, color='red')

    # Plot formatting
    plt.xlabel(f"Time ({time_unit_name})", fontsize=12)
    plt.ylabel(f"{attribute_name}", fontsize=12)
    metrics_str = ""
    if metrics and not all(np.isnan(list(metrics.values()))):
         r2_str = f"{metrics.get('R2', np.nan):.3f}" if not np.isnan(metrics.get('R2', np.nan)) else "N/A"
         rmse_str = f"{metrics.get('RMSE', np.nan):.3f}" if not np.isnan(metrics.get('RMSE', np.nan)) else "N/A"
         metrics_str = f" (R2={r2_str}, RMSE={rmse_str})"
    plt.title(f"Model Fit for: {attribute_name}{metrics_str}\n(Trained up to {train_limit_time} {time_unit_name})", fontsize=14)
    plt.axvline(train_limit_time, color='grey', linestyle=':', linewidth=1.5, label=f'Training End ({train_limit_time} {time_unit_name})')

    # Simplified Legend Creation
    handles, labels = plt.gca().get_legend_handles_labels()
    filled_marker_handle = plt.scatter([],[], marker='o', color='grey', label='_nolegend_')
    hollow_marker_handle = plt.scatter([],[], marker='o', facecolor='none', edgecolor='grey', label='_nolegend_')
    unique_labels_dict = {}; batch_handles = {}; fit_handles = {}; other_handles = {}
    for h, l in zip(handles, labels):
        if l == '_nolegend_' or ' Train' in l or ' Test' in l: continue
        if l not in unique_labels_dict:
            unique_labels_dict[l] = h
            if 'Batch' in l: batch_handles[l] = h
            elif 'Fit' in l: fit_handles[l] = h
            elif 'Training End' in l: other_handles[l] = h
    ordered_handles = list(batch_handles.values()) + [filled_marker_handle, hollow_marker_handle] + list(fit_handles.values()) + list(other_handles.values())
    ordered_labels = list(batch_handles.keys()) + ['Train Data (Filled)', 'Test Data (Hollow)'] + list(fit_handles.keys()) + list(other_handles.keys())
    legend = plt.legend(handles=ordered_handles, labels=ordered_labels, bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=9, title="Legend", title_fontsize='10')
    filled_marker_handle.remove(); hollow_marker_handle.remove()

    # Remove Grid
    plt.grid(False) # Explicitly disable grid

    # Adjust Y-axis limits based on combined data and prediction range
    data_range = max_plot_lim - min_plot_lim
    if data_range < 1e-6: data_range = max(1.0, abs(max_plot_lim * 0.1)) # Avoid zero range
    padding = 0.05 * data_range # 5% padding

    final_min_y = min_plot_lim - padding
    final_max_y = max_plot_lim + padding
    if min_plot_lim >= 0: final_min_y = max(0, final_min_y) # Ensure y-min >= 0 if data is non-negative

    plt.ylim(final_min_y, final_max_y)
    plt.xlim(left=-max(extrapolate_limit_time*0.02, 0.5)) # Small negative padding on x-axis

    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust layout for legend
    plt.show()


# --- Main Execution Loop ---
if __name__ == '__main__':
    time_unit_label = TIME_COLUMN.split('(')[-1].split(')')[0] if '(' in TIME_COLUMN else 'Units'
    results_summary_all_attributes = {}
    for attribute_name in ATTRIBUTES_TO_MODEL:
        print("\n" + "="*20 + f" Processing Attribute: {attribute_name} " + "="*20)
        # 1. Prepare Data
        cols_to_keep = ['T_c', 'time', 'batch', 'T_k', attribute_name]
        data_df_current_attr = data_df_full[cols_to_keep].copy()
        data_df_current_attr.rename(columns={attribute_name: 'concentration'}, inplace=True)
        data_df_current_attr['concentration'] = pd.to_numeric(data_df_current_attr['concentration'], errors='coerce')
        initial_attr_rows = len(data_df_current_attr)
        data_df_current_attr.dropna(subset=['concentration'], inplace=True)
        rows_dropped = initial_attr_rows - len(data_df_current_attr)
        if rows_dropped > 0: print(f"INFO: Dropped {rows_dropped} rows missing '{attribute_name}'.")
        if data_df_current_attr.empty:
            print(f"WARNING: No valid data for '{attribute_name}'. Skipping."); results_summary_all_attributes[attribute_name] = {'fit_successful': False, 'metrics': {}, 'params': {}, 'num_params': 0, 'error': 'No valid data'}; continue

        # 2. Determine Initial Conditions
        time_zero_data = data_df_current_attr[data_df_current_attr['time'] == 0]['concentration']
        current_settings = attribute_settings.get(attribute_name, {})
        y_scale = current_settings.get('y_scale', 100.0)
        if 'initial_sign' in current_settings:
            initial_sign = current_settings['initial_sign']; print(f"Using specified initial_sign: {initial_sign} for '{attribute_name}'")
        else:
            print(f"Attempting heuristic initial_sign for '{attribute_name}' based on initial slope...")
            all_initial_slopes = []
            unique_temps = data_df_current_attr['T_k'].unique()
            min_points_for_slope = 2; num_initial_points_to_use = 3
            for T_k in unique_temps:
                temp_df = data_df_current_attr[data_df_current_attr['T_k'] == T_k].sort_values('time')
                grouped_time = temp_df.groupby('time')['concentration'].mean().reset_index()
                if len(grouped_time) >= min_points_for_slope:
                    initial_data = grouped_time.head(num_initial_points_to_use)
                    if len(initial_data) >= min_points_for_slope:
                        times = initial_data['time'].astype(float).values; concentrations = initial_data['concentration'].astype(float).values
                        if np.ptp(times) > 1e-6:
                           try: slope, _ = np.polyfit(times, concentrations, 1); all_initial_slopes.append(slope)
                           except Exception as e: print(f"  WARN ({attribute_name}): Slope calc error T={T_k-273.15:.1f}C: {e}.")
                        elif len(initial_data) > 1:
                             diff = concentrations[-1] - concentrations[0]
                             if abs(diff) > 1e-6: all_initial_slopes.append(np.sign(diff)); print(f"  INFO ({attribute_name}): Using sign diff T={T_k-273.15:.1f}C.")
            if not all_initial_slopes: print(f"  WARN ({attribute_name}): No slope found. Defaulting to -1."); initial_sign = -1.0
            else:
                median_slope = np.median(all_initial_slopes); print(f"  Median initial slope ({len(all_initial_slopes)} temps): {median_slope:.4g}")
                if abs(median_slope) < 1e-6: print(f"  WARN ({attribute_name}): Slope near zero. Defaulting to -1."); initial_sign = -1.0
                else: initial_sign = 1.0 if median_slope > 0 else -1.0
            print(f"Using heuristic initial_sign: {initial_sign} for '{attribute_name}'")
        likely_formation = initial_sign > 0
        if time_zero_data.empty: y_initial = 0.0 if likely_formation else y_scale; print(f"WARNING: No t=0 data for '{attribute_name}'. Default y0={y_initial:.3f}")
        else: y_initial = time_zero_data.mean(); print(f"Determined y0 for '{attribute_name}': {y_initial:.3f}")
        if not time_zero_data.empty and time_zero_data.std() > 0.05 * y_scale: print(f"WARNING: High std dev ({time_zero_data.std():.2f}) in initial values for '{attribute_name}'.")
        y0_tensor = torch.tensor([y_initial], dtype=torch.float32)

        # 3. Split Data
        train_df = data_df_current_attr[data_df_current_attr['time'] <= TRAIN_UP_TO_TIME_UNIT].copy()
        if train_df.empty: print(f"WARNING: No training data for '{attribute_name}'. Skipping."); results_summary_all_attributes[attribute_name] = {'fit_successful': False, 'metrics': {}, 'params': {}, 'num_params': 0, 'error': 'No training data'}; continue
        print(f"Using {len(train_df)} points for training '{attribute_name}' (<= {TRAIN_UP_TO_TIME_UNIT} {time_unit_label}).")
        train_data_grouped_by_temp = {}
        T_kelvin_train = np.sort(train_df['T_k'].unique())
        for T_k in T_kelvin_train:
            temp_df = train_df[train_df['T_k'] == T_k]
            if not temp_df.empty: train_data_grouped_by_temp[T_k] = {'time': torch.tensor(temp_df['time'].values, dtype=torch.float32), 'concentration': torch.tensor(temp_df['concentration'].values, dtype=torch.float32)}
        print(f"Training data prepared for {len(train_data_grouped_by_temp)} temps for '{attribute_name}'."); print("-" * 30)

        # 4. Instantiate and Train Model
        model_name_attr = f"UDE_{attribute_name.replace(' (%)', '').replace(' ', '_').replace('/', '_')}"
        # Note: NN complexity increased, weight decay kept, LR schedule adjusted
        ude_model_instance = UDEModel(initial_logA=np.log(1e10), initial_logEa=np.log(75000), initial_log_n=np.log(1.0), initial_log_m=np.log(1.0), initial_sign=initial_sign, y_scale=y_scale)
        current_num_params = ude_model_instance.get_num_params(); print(f"Instantiated {model_name_attr} ({current_num_params} params).")
        ude_losses, fit_successful = train_ude_parameters(ude_model_instance, train_data_grouped_by_temp, y0_tensor, model_name=model_name_attr, epochs=6000, lr=0.001) # Using weight decay & new schedule

        # 5. Evaluate Model
        print(f"\n--- Goodness of Fit for {model_name_attr} (Training Data) ---")
        metrics = {'R2': np.nan, 'RMSE': np.nan, 'AIC': np.nan, 'BIC': np.nan}; final_param_values = {}
        if fit_successful:
            final_param_values = ude_model_instance.get_params_dict()
            y_actual_train, y_pred_train = get_predictions_and_actuals(ude_model_instance, train_df, y0_tensor)
            if len(y_actual_train) > 0 and len(y_pred_train) == len(y_actual_train): metrics = calculate_fit_metrics(y_actual_train, y_pred_train, len(y_actual_train), current_num_params, model_name_attr)
            else: print(f"WARNING ({model_name_attr}): No valid train preds for metrics.")
        else: print(f"Fit failed for {model_name_attr}, skip metrics."); final_param_values = {p: np.nan for p in ['A', 'Ea', 'n', 'm', 'sign']}

        # 6. Store Results
        results_summary_all_attributes[attribute_name] = {'params': final_param_values, 'metrics': metrics, 'fit_successful': fit_successful, 'num_params': current_num_params, 'model_name': model_name_attr, 'error': None if fit_successful else 'Fit Failed'}

        # 7. Plot Results
        try:
            plot_results(attribute_name, ude_model_instance, data_df_current_attr, y0_tensor, TRAIN_UP_TO_TIME_UNIT, EXTRAPOLATE_TO_TIME_UNIT, metrics, fit_successful, time_unit_label)
        except Exception as plot_err: print(f"ERROR generating plot for {attribute_name}: {plot_err}"); results_summary_all_attributes[attribute_name]['error'] = 'Plotting Error'

        # Optional: Plot Training Loss Curve
        if ude_losses and not all(np.isnan(ude_losses)):
             plt.figure(figsize=(8, 5)); plt.plot(ude_losses, label=f'{model_name_attr} Loss', color='green')
             plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.yscale('log'); plt.title(f'Loss Curve ({model_name_attr})'); plt.legend(); plt.tight_layout(); plt.show()

    # --- Final Summary Table ---
    print("\n" + "="*30 + " FINAL MODEL SUMMARY (All Attributes) " + "="*30)
    summary_list = []
    for attribute, results in results_summary_all_attributes.items():
        row = {'Attribute': attribute, 'Model Name': results.get('model_name', 'N/A'), 'Fit Successful': results.get('fit_successful', False), 'Num Params': results.get('num_params', 0), 'Error Status': results.get('error', '')}
        metrics = results.get('metrics', {}); row['R2'] = metrics.get('R2', np.nan); row['RMSE'] = metrics.get('RMSE', np.nan); row['AIC'] = metrics.get('AIC', np.nan); row['BIC'] = metrics.get('BIC', np.nan)
        params = results.get('params', {})
        if isinstance(params, dict) and params and results.get('fit_successful'): row['Parameters'] = ", ".join([f"{k}={v:.3g}" if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else f"{k}=NaN/Inf" for k,v in params.items()])
        elif not results.get('fit_successful'): row['Parameters'] = "Fit Failed"
        else: row['Parameters'] = "N/A"
        summary_list.append(row)
    summary_df_final = pd.DataFrame(summary_list)
    cols_order_final = ['Attribute', 'Model Name', 'Fit Successful', 'Num Params', 'R2', 'RMSE', 'AIC', 'BIC', 'Parameters', 'Error Status']
    for col in cols_order_final:
        if col not in summary_df_final.columns:
             if col in ['Attribute', 'Model Name', 'Parameters', 'Error Status']: summary_df_final[col] = ''
             elif col == 'Fit Successful': summary_df_final[col] = False
             elif col == 'Num Params': summary_df_final[col] = 0
             else: summary_df_final[col] = np.nan
    summary_df_final = summary_df_final[cols_order_final]
    print(summary_df_final.to_string(index=False, float_format="%.4f", na_rep='NaN'))
    print("\nLower AIC/BIC indicate better balance between fit quality and model complexity.")
    print("Check R2 (closer to 1), RMSE (lower), and visual fit on the individual plots.")
    print(f"Time unit used for training/plotting: {time_unit_label}")