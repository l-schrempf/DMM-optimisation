# This code optimises the parameters to an LTSpice model to match experimental data
import math
import time
import argparse
import toml
import numpy as np
import ltspice
from PyLTSpice import SimRunner
from PyLTSpice import SpiceEditor
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution

# Process CLI args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", type=str, help='Config file', required=True)
args = parser.parse_args()

# Load configuration file
config = toml.load(args.config_file)
fixed = config['fixed']                    # Fixed parameters dict.
params_init = config['variable']           # Variable parameters dict. to be optimized
param_names = list(params_init.keys())
param_bounds = config['bounds']            # Variable parameter bounds, i.e. (min, max)-tuples
param_bound_types = config['bound_types']  # Bound types, one of "geom" or "lin"

# Select spice model
LTC = SimRunner(output_folder='temp', parallel_sims=16)
print("Sim. runner created...")
netlist = SpiceEditor('../Draft6.net')
print("Netlist editor created...")

# Define voltage sweep
Vp = config['Vp']
Vm = config['Vm']
t_sweep_minus = config['t_sweep_minus']
t_sweep_plus = config['t_sweep_plus']
t_wait = config['t_wait']
t_tot = t_sweep_minus + t_sweep_plus + t_wait
netlist.set_element_model('V1', f'PWL(0 0 {0.5 * t_sweep_plus} {Vp} {t_sweep_plus} 0 {t_wait + t_sweep_plus} 0 {t_wait + t_sweep_plus + 0.5 * t_sweep_minus} {Vm} {t_tot} 0)')
netlist.add_instructions(f".tran 0 {t_tot} 0 0.0001",)

# Set fixed parameters
netlist.set_parameters(**fixed)

# Load experimental data
i_meas_set, v_meas_set = np.loadtxt(config['data_set'], delimiter='\t').T
i_meas_reset, v_meas_reset = np.loadtxt(config['data_reset'], delimiter='\t').T
N_set = i_meas_set.shape[0]
N_reset = i_meas_reset.shape[0]
i_meas = np.concatenate([i_meas_set, i_meas_reset])
v_meas = np.concatenate([v_meas_set, v_meas_reset])
print(i_meas)
print(v_meas)
i_meas_max = v_meas.argmax()
i_meas_min = v_meas.argmin()

# Function that takes in a list of parameter dicts and outputs V-I for each set of params
def run_simulations(params_dicts):
    N = len(params_dicts)
    
    # Start all tasks and store their IDs
    ids = np.empty(N, dtype=int)
    for i in range(N):
        # Set the netlist parameters
        netlist.set_parameters(**params_dicts[i])
        print("Running simulation")
        task = LTC.run(netlist)
        ids[i] = task.runno

    # Wait to finish simulations
    did_succeed = LTC.wait_completion()
    if not did_succeed:
        raise RuntimeError("Simulation failed")
    print("Simulations finished")
    # time.sleep(1)

    # Read simulated data from LTSpice raw file
    res = []
    for i in range(N):
        file_path = f'./temp/Draft6_{ids[i]}.raw'
        ltr = ltspice.Ltspice(file_path)
        ltr.parse()
        voltage_in = ltr.get_data('V(+)')
        current = ltr.get_data('Ix(dmm:+)')
        res.append((voltage_in, np.abs(current)))
        # plt.plot(time, voltage_in)
        # plt.plot(time, current)
        # plt.show()

    return res

def calc_mse(params_list):
    N, S = params_list.shape
    params_dicts = []
    for p in params_list.T:
        pd = dict(zip(param_names, p))
        params_dicts.append(pd)
        print(pd)
    res = run_simulations(params_dicts)
    mses = np.empty(S)
    for n, r in enumerate(res):
        v, i = r
        i_max = v.argmax()
        i_min = v.argmin()

        i_sim = np.zeros_like(i_meas)

        f_1 = interpolate.interp1d(v[:i_max], i[:i_max], fill_value="extrapolate")
        i_sim[:i_meas_max] = f_1(v_meas[:i_meas_max])

        f_2 = interpolate.interp1d(v[i_max:i_min], i[i_max:i_min], fill_value="extrapolate")
        i_sim[i_meas_max:i_meas_min] = f_2(v_meas[i_meas_max:i_meas_min])

        f_3 = interpolate.interp1d(v[i_min:], i[i_min:], fill_value="extrapolate")
        i_sim[i_meas_min:] = f_3(v_meas[i_meas_min:])

        mse = ((np.log(np.abs(i_meas)) - np.log(np.abs(i_sim))) ** 2).mean()
        mses[n] = mse
        print('MSE =', mse)
    print('------------------------')
    return mses

# Plot init. fit
v, i = run_simulations([params_init])[0]
palette = sns.color_palette("coolwarm", 2) # bright or Spectral_r
plt.scatter(v_meas, i_meas, 15, c='k', label="Measured")
plt.plot(v, i, c='salmon', linewidth=2.5, label="Simulated")
plt.yscale('log')
plt.ylabel("Current (A)")
plt.xlabel("Voltage (V)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("figures/" + args.config_file[:-5] + "_init.png", dpi=200)
plt.show()

# Plot param. ranges
num_cols = 4
num_rows = math.ceil(len(param_bounds.keys()) / num_cols)
num_vals = 5
palette = sns.color_palette("coolwarm", num_vals) # bright or Spectral_r
fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
for i, item in enumerate(param_bounds.items()):
    k, v = item
    r = i // num_cols
    c = i - r * num_cols
    axs[r, c].scatter(v_meas, i_meas, 15, 'k', label="Measured")
    if list(param_bounds.items())[i] == "geom":
        xs = np.geomspace(*v, num_vals)
    else:
        xs = np.linspace(*v, num_vals)

    params_dicts = []
    for x in xs:
        pd = params_init.copy()
        pd[k] = x
        params_dicts.append(pd)
    res = run_simulations(params_dicts)
    for i in range(num_vals):
        v, cur = res[i]
        color = palette[i]
        axs[r, c].plot(v, cur, c=color, label=f"{k}={xs[i]:.1e}")
    axs[r, c].set_ylabel("Current (A)")
    axs[r, c].set_xlabel("Voltage (V)")
    axs[r, c].set_yscale('log')
    axs[r, c].legend(loc="lower left")
for i in range(c + 1, num_cols): # remove remaining plots
    fig.delaxes(axs[r, i])
fig.tight_layout()
plt.savefig("figures/" + args.config_file[:-5] + "_sweep.png", dpi=200)
plt.show()

# Fit parameters
# opt = minimize(calc_mse, x0=list(params_init.values()), method="Nelder-Mead", options={"maxiter": config['maxiter']})
opt = differential_evolution(calc_mse, x0=list(params_init.values()),
                             bounds=list(param_bounds.values()), maxiter=config['maxiter'],
                             vectorized=True, polish=False)
params_opt = dict(zip(param_names, opt['x']))
print(params_opt)

# Plot fit
plt.plot(v_meas, i_meas, label="GT")
v, i = run_simulations([params_init])[0]
plt.plot(v, i, label="Init")
v, i = run_simulations([params_opt])[0]
print(v, i)
plt.plot(v, i, label="Opt")
plt.yscale('log')
plt.ylabel('Current (A)')
plt.xlabel('Voltage (V)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/' + args.config_file[:-5] + '_opt.png', dpi=200)
plt.show()