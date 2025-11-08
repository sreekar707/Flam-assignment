# fit_curve.py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ---------- Model ----------
def model_xy(t, theta, M, X):
    """Return x(t), y(t) for arrays t. theta in radians."""
    t = np.asarray(t)
    exp_term = np.exp(M * np.abs(t))     # for t>=0, same as exp(M*t)
    s = np.sin(0.3 * t)
    x = t * np.cos(theta) - exp_term * s * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + exp_term * s * np.cos(theta)
    return x, y

# ---------- Residuals (unknown t_i case) ----------
def residuals_unknown_t(all_vars, x_obs, y_obs, n_points):
    """
    all_vars: [theta, M, X, t1, t2, ..., tn]
    Returns stacked residuals [x_pred - x_obs, y_pred - y_obs]
    """
    theta = all_vars[0]
    M = all_vars[1]
    X = all_vars[2]
    t_vals = all_vars[3:]
    x_pred, y_pred = model_xy(t_vals, theta, M, X)
    r = np.empty(n_points * 2)
    r[0::2] = x_pred - x_obs
    r[1::2] = y_pred - y_obs
    return r

# ---------- Fit routine ----------
def fit_xy_only(filepath, plot=True):
    df = pd.read_csv(filepath)
    x = df['x'].values
    y = df['y'].values
    n = x.size

    # Initial guesses:
    theta0 = np.deg2rad(25.0)        # 25 degrees in radians
    M0 = 0.0
    X0 = x.mean()                    # shift start guess = mean of x
    t0s = np.linspace(6.0, 60.0, n)  # initial t_i guesses spread across the domain

    all0 = np.concatenate(([theta0, M0, X0], t0s))

    # Bounds:
    lb = np.concatenate(([0.0, -0.05, 0.0], np.full(n, 6.0)))
    ub = np.concatenate(([np.deg2rad(50.0), 0.05, 100.0], np.full(n, 60.0)))

    print("Starting optimization. This may take a bit (few seconds to minutes).")
    res = least_squares(residuals_unknown_t, all0, bounds=(lb, ub),
                        args=(x, y, n), max_nfev=200000, xtol=1e-9, ftol=1e-9)

    theta_hat = res.x[0]
    M_hat = res.x[1]
    X_hat = res.x[2]
    t_hats = res.x[3:]

    # Diagnostics
    print("Success:", res.success, res.message)
    print("Estimated theta (deg):", np.rad2deg(theta_hat))
    print("Estimated M:", M_hat)
    print("Estimated X:", X_hat)

    # Compute fitted curve for a smooth t grid
    t_grid = np.linspace(6, 60, 1000)
    x_fit, y_fit = model_xy(t_grid, theta_hat, M_hat, X_hat)

    # Plot data and fitted curve
    if plot:
        plt.figure(figsize=(8,6))
        plt.scatter(x, y, s=8, alpha=0.6, label='data (x,y)')
        plt.plot(x_fit, y_fit, '-', linewidth=2, label='fitted parametric curve')
        # optional: show the predicted points for the estimated t_i
        x_pred_pts, y_pred_pts = model_xy(t_hats, theta_hat, M_hat, X_hat)
        plt.scatter(x_pred_pts, y_pred_pts, s=10, facecolors='none', edgecolors='r', label='fitted pts (t_i)')
        plt.legend()
        plt.xlabel('x'); plt.ylabel('y'); plt.title('Data vs fitted curve')
        plt.axis('equal')
        plt.show()

    # L1 error (uniform t sample)
    x_uniform, y_uniform = model_xy(t_grid, theta_hat, M_hat, X_hat)
    # If you want L1 distance between data and curve, we compute average L1 distance
    # from each data point to its fitted point (using estimated t_i)
    l1_per_point = np.abs(x_pred_pts - x) + np.abs(y_pred_pts - y)
    mean_l1 = np.mean(l1_per_point)
    print("Mean L1 per point (using estimated t_i):", mean_l1)

    return {'theta_rad': theta_hat, 'theta_deg': np.rad2deg(theta_hat),
            'M': M_hat, 'X': X_hat, 't_hats': t_hats, 'res': res}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fit_curve.py path/to/xy_data.csv")
    else:
        out = fit_xy_only(sys.argv[1], plot=True)
        print("\nFinal formula (LaTeX-ready):")
        th = out['theta_rad']; M = out['M']; X = out['X']
        print(r"(t*\cos({:.6f}) - e^{{{:.6f} |t|}} \sin(0.3 t) \sin({:.6f}) + {:.6f}, 42 + t*\sin({:.6f}) + e^{{{:.6f}|t|}} \sin(0.3 t) \cos({:.6f}))".format(th, M, th, X, th, M, th))
