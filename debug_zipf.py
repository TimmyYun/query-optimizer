import numpy as np
import matplotlib.pyplot as plt
from datasets import gen_values
from models import HybridEstimator, Bucket, RangeQuery
import seaborn as sns

def debug_zipf():
    print("Generating Zipf Data...")
    N = 1000000
    rng = np.random.default_rng(42)
    dataset = gen_values(rng, "zipf", N, 0, 200000)
    mn, mx = dataset.min(), dataset.max()
    print(f"Range: [{mn}, {mx}]")

    # 1. Create Equi-Width Buckets (Simulation)
    n_buckets = 100
    width = (mx - mn + 1) / n_buckets
    buckets = []
    
    # Simple fixed width buckets
    curr = mn
    for i in range(n_buckets):
        lo = int(curr)
        hi = int(curr + width - 1)
        if i == n_buckets - 1: hi = mx
        
        # Count items
        mask = (dataset >= lo) & (dataset <= hi)
        count = np.sum(mask)
        buckets.append(Bucket(lo, hi, count))
        curr += width

    print(f"Top 5 Buckets by Count:")
    for i in range(5):
        print(f"Bucket {i}: [{buckets[i].lo}, {buckets[i].hi}] Count={buckets[i].count}")

    # 2. Train Hybrid Estimator
    print("\nTraining Hybrid Estimator...")
    hybrid = HybridEstimator(buckets, identity_threshold=0.0001) # Low threshold to force learning
    
    # Calculate Frequency for training
    freqs = np.bincount(dataset - mn, minlength=mx-mn+1)
    
    hybrid.train(freqs, mn, points_per_bucket=50, rng=np.random.default_rng(42))
    
    # 3. Inspect Model Selection for Top Buckets
    print("\nModel Selection Analysis:")
    
    # Enable debug printing in models.py via monkeypatch or just by checking the stored model
    # To see MSEs, we'd need to modify models.py, but let's first just infer from what we have.
    # Actually, we can manually run the candidates here to see their MSEs.
    
    b = buckets[0]
    print(f"Analyzing Bucket 0: [{b.lo}, {b.hi}] Count={b.count}")
    
    # Re-create training/val data as models.py does
    freq = freqs
    ps = np.cumsum(freq)
    mn = dataset.min()
    
    # ... (Copying sampling logic snippet for debug) ...
    # Simplified for brevity, just getting actual data
    data_in_bucket = dataset[(dataset >= b.lo) & (dataset <= b.hi)]
    y_actual = np.arange(1, len(data_in_bucket) + 1) / len(data_in_bucket)
    x_norm = (np.sort(data_in_bucket) - b.lo) / (b.hi - b.lo + 1)
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    
    X = x_norm.reshape(-1, 1)
    y = y_actual
    
    # Linear
    m_lin = Ridge().fit(X, y)
    mse_lin = np.mean((y - m_lin.predict(X))**2)
    print(f"Linear MSE: {mse_lin:.6f}")
    
    # Log-Linear (Old)
    # Try with very low alpha to avoid underfitting the slope
    X_log = np.log(X + 1e-7)
    m_log = Ridge(alpha=1e-5).fit(X_log, y) # Reduced alpha
    mse_log = np.mean((y - m_log.predict(X_log))**2)
    print(f"Log-Linear (Old, alpha=1e-5) MSE: {mse_log:.6f}")
    print(f"  Coef: {m_log.coef_[0]}, Intercept: {m_log.intercept_}")
    
    # Log-Linear (New - Correct Bias)
    width = b.hi - b.lo + 1
    bias = 1.0 / width
    X_log_new = np.log(X + bias)
    m_log_new = Ridge(alpha=1e-5).fit(X_log_new, y)
    mse_log_new = np.mean((y - m_log_new.predict(X_log_new))**2)
    print(f"Log-Linear (New, alpha=1e-5) MSE: {mse_log_new:.6f}")
    print(f"  Coef: {m_log_new.coef_[0]}, Intercept: {m_log_new.intercept_}")
    
    # Poly
    # ... (keep poly)
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    m_poly = Ridge().fit(X_poly, y)
    mse_poly = np.mean((y - m_poly.predict(X_poly))**2)
    print(f"Poly MSE: {mse_poly:.6f}")
    
    # Current selection
    model = hybrid.models.get(0)
    print(f"Selected Model: {model}")
        
    # 4. Plot Actual CDF vs Model CDF for Bucket 0
    # Model Preds
    x_samples = np.linspace(0, 1, 100)
    y_preds = [hybrid._predict_local_cdf(model, x) for x in x_samples]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_norm, y_actual, label="Actual CDF", color='black', linewidth=2)
    plt.plot(x_samples, y_preds, label=f"Model (Selected)", color='red', linestyle='--')
    plt.title(f"Bucket 0 CDF: Actual vs Model (Zipf)\nRange: [{b.lo}, {b.hi}]")
    plt.legend()
    plt.grid(True)
    plt.savefig("debug_zipf_bucket_0.png")
    print("Saved debug_zipf_bucket_0.png")
    
    # 5. Q-Error Evaluation on Bucket 0
    print("\nEvaluated Q-Error on Bucket 0:")
    # Generate random queries within bucket 0
    q_errors = []
    n_queries = 1000
    queries = []
    # ... (previous code)
    
    # Define a helper to evaluate Q-Error for a given model (or tuple)
    def eval_model_q_error(model_cand, name):
         q_errs = []
         dummy_hybrid = HybridEstimator([], 1e-5) # Helper to access _predict_local_cdf
         
         for q in queries:
             # Actual
             mask = (dataset >= q.low) & (dataset <= q.high)
             act_count = np.sum(mask)
             act = max(1, act_count) # Use 1 as min for Q-Error denominator usually
             
             # Pred
             lo_norm = (q.low - b.lo) / (b.hi - b.lo + 1)
             hi_norm = (q.high - b.lo) / (b.hi - b.lo + 1)
             
             # Need to handle tuple vs object manually since _predict_local_cdf expects stored format
             # actually _predict_local_cdf takes 'model' which can be either.
             # But here we have sklearn models for Lin/Log/Poly.
             # We need to wrap them into the format Hybrid uses.
             
             if name == "Linear":
                 # sklearn Ridge
                 # Hybrid expects ('linear', coef, intercept)
                 m_fmt = ("linear", model_cand.coef_[0], model_cand.intercept_)
                 val_hi = dummy_hybrid._predict_local_cdf(m_fmt, hi_norm)
                 val_lo = dummy_hybrid._predict_local_cdf(m_fmt, lo_norm)
             elif name == "Log-Linear":
                  m_fmt = ("log_linear", model_cand.coef_[0], model_cand.intercept_)
                  val_hi = dummy_hybrid._predict_local_cdf(m_fmt, hi_norm)
                  val_lo = dummy_hybrid._predict_local_cdf(m_fmt, lo_norm)
             elif name == "Log-Linear (New)":
                  # Manual prediction because dummy_hybrid uses old feature transform
                  # Model: y = a * log(x + bias) + b
                  # We need to use the SAME bias as training
                  bias = 1.0 / (b.hi - b.lo + 1)
                  
                  # Predict Hi
                  feat_hi = np.log(hi_norm + bias)
                  val_hi = model_cand.predict([[feat_hi]])[0]
                  val_hi = np.clip(val_hi, 0, 1)
                  
                  # Predict Lo
                  feat_lo = np.log(lo_norm + bias)
                  val_lo = model_cand.predict([[feat_lo]])[0]
                  val_lo = np.clip(val_lo, 0, 1)

             elif name == "Poly":
                  m_fmt = ("poly", model_cand.coef_, model_cand.intercept_)
                  val_hi = dummy_hybrid._predict_local_cdf(m_fmt, hi_norm)
                  val_lo = dummy_hybrid._predict_local_cdf(m_fmt, lo_norm)
             else:
                  # FourierMLP (already correct format)
                  val_hi = dummy_hybrid._predict_local_cdf(model_cand, hi_norm)
                  val_lo = dummy_hybrid._predict_local_cdf(model_cand, lo_norm)
                  
             pred_count = (val_hi - val_lo) * b.count
             pred = max(1, pred_count)
             
             q_err = max(act/pred, pred/act)
             q_errs.append(q_err)
         
         print(f"{name} Median Q-Error: {np.median(q_errs):.4f}")
         return q_errs

    # Generate queries once
    queries = []
    for _ in range(1000):
        l = rng.integers(b.lo, b.hi)
        r = rng.integers(l, b.hi + 1)
        queries.append(RangeQuery(l, r))
        
    print("\nComparing Models on Q-Error:")
    eval_model_q_error(m_lin, "Linear")
    eval_model_q_error(m_log, "Log-Linear")
    eval_model_q_error(m_log_new, "Log-Linear (New)")
    eval_model_q_error(m_poly, "Poly")
    eval_model_q_error(model, "Selected(Fourier)")

if __name__ == "__main__":
    debug_zipf()
