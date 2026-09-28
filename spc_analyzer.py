import numpy as np
import matplotlib.pyplot as plt

def calculate_spc_limits(data, A2=0.577, D3=0, D4=2.114):
    """
    Alt grup büyüklüğü n=5 için standart SPC katsayıları ile
    X-bar ve R kontrol sınırlarını hesaplar.
    data: (m, n) boyutunda numpy dizisi (m örneklem, n alt grup boyutu)
    """
    x_bar = np.mean(data, axis=1)
    r = np.ptp(data, axis=1)  # Peak-to-peak (Max - Min)
    
    cl_x = np.mean(x_bar)
    cl_r = np.mean(r)
    
    ucl_x = cl_x + A2 * cl_r
    lcl_x = cl_x - A2 * cl_r
    
    ucl_r = D4 * cl_r
    lcl_r = D3 * cl_r
    
    return x_bar, r, (cl_x, ucl_x, lcl_x), (cl_r, ucl_r, lcl_r)

def calculate_capability(data, lsl, usl, d2=2.326):
    """Cp ve Cpk süreç yeterlilik indekslerini hesaplar."""
    overall_mean = np.mean(data)
    mean_r = np.mean(np.ptp(data, axis=1))
    sigma_within = mean_r / d2  # Süreç içi standart sapma tahmini
    
    cp = (usl - lsl) / (6 * sigma_within)
    cpk = min((usl - overall_mean) / (3 * sigma_within), 
              (overall_mean - lsl) / (3 * sigma_within))
    
    return cp, cpk, overall_mean, sigma_within

def plot_spc_charts(x_bar, r, x_limits, r_limits, cp, cpk):
    """X-bar ve R kartlarını tek bir görselde çizer ve kaydeder."""
    cl_x, ucl_x, lcl_x = x_limits
    cl_r, ucl_r, lcl_r = r_limits
    samples = np.arange(1, len(x_bar) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    
    # 1. X-bar Grafiği
    ax1.plot(samples, x_bar, marker='o', color='#2563eb', linewidth=1.5, label='Alt Grup Ortalaması')
    ax1.axhline(cl_x, color='#16a34a', linestyle='--', label=f'CL (Merkez): {cl_x:.2f}')
    ax1.axhline(ucl_x, color='#dc2626', linestyle='-', label=f'UCL: {ucl_x:.2f}')
    ax1.axhline(lcl_x, color='#dc2626', linestyle='-', label=f'LCL: {lcl_x:.2f}')
    ax1.set_title(f"X-Bar Kontrol Kartı | Cp: {cp:.2f}, Cpk: {cpk:.2f}", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Ölçüm Ortalaması (mm)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. R Grafiği
    ax2.plot(samples, r, marker='s', color='#7c3aed', linewidth=1.5, label='Aralık (Range - R)')
    ax2.axhline(cl_r, color='#16a34a', linestyle='--', label=f'CL: {cl_r:.2f}')
    ax2.axhline(ucl_r, color='#dc2626', linestyle='-', label=f'UCL: {ucl_r:.2f}')
    ax2.axhline(lcl_r, color='#dc2626', linestyle='-', label=f'LCL: {lcl_r:.2f}')
    ax2.set_title("R (Değişkenlik/Aralık) Kontrol Kartı", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Alt Grup Örnek No")
    ax2.set_ylabel("Aralık Değeri (R)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("spc_control_chart.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    # 20 alt grup, her birinde 5 ölçüm (n=5)
    sample_measurements = np.random.normal(loc=50.0, scale=0.3, size=(20, 5))
    LSL, USL = 49.0, 51.0  # Müşteri spesifikasyon limitleri
    
    x_bar, r, x_lim, r_lim = calculate_spc_limits(sample_measurements)
    cp, cpk, mu, sigma = calculate_capability(sample_measurements, LSL, USL)
    
    print(f"Süreç Analizi Sonucu -> Cp: {cp:.3f}, Cpk: {cpk:.3f}")
    plot_spc_charts(x_bar, r, x_lim, r_lim, cp, cpk)
