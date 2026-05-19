import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import (
    load_image, load_watermark,
    embed_lsb, extract_lsb,
    calculate_psnr, calculate_ber,
    show_watermark_comparison, plot_metrics_table
)

IMAGE_PATH     = 'images/face.jpg'
WATERMARK_PATH = 'images/logo.png'
WM_SIZE        = (64, 64)
LOCATION       = 'top-left'
QUALITY_FACTORS = [100, 90, 80, 70, 50, 30, 10]

os.makedirs('results/watermarked', exist_ok=True)
os.makedirs('results/compressed', exist_ok=True)
os.makedirs('results/extracted/qf_experiment', exist_ok=True)

print("=" * 65)
print("RUNNING: EKSPERIMEN Pengaruh Quality Factor JPEG")
print("=" * 65)

image = load_image(IMAGE_PATH, mode='gray')

watermark = load_watermark(WATERMARK_PATH, WM_SIZE)

watermarked = embed_lsb(image, watermark, location=LOCATION)
wm_png_path = 'results/watermarked/watermarked_BASE.png'
cv2.imwrite(wm_png_path, watermarked)

extracted_verify = extract_lsb(watermarked, WM_SIZE, location=LOCATION)
ber_verify = calculate_ber(watermark, extracted_verify)
print(f"[VERIFIKASI] BER In-Memory Array: {ber_verify}")
if ber_verify != 0.0:
    print("Deteksi kegagalan fungsi dasar LSB!")
    exit()
print("Memulai proses kompresi matematika...\n")

def simulate_jpeg_compression_manual(image, qf):
    img = image.copy().astype(np.float64)
    h, w = img.shape[:2]
    
    if qf >= 50:
        scale = (100 - qf) / 50
    else:
        scale = 50 / qf
        
    if scale == 0:
        return image.copy() 
        
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            r_end = min(r + 8, h)
            c_end = min(c + 8, w)
            
            block = img[r:r_end, c:c_end]
            block_mean = np.mean(block)
            
            compressed_block = block_mean + (block - block_mean) / (1 + scale * 0.45)
            img[r:r_end, c:c_end] = compressed_block
            
    return np.clip(img, 0, 255).astype(np.uint8)

results = []
extracted_wms = []
qf_labels = []

print(f"{'QF':>5} | {'PSNR Img':>10} | {'BER':>8} | {'Error Bits':>11} | {'Est. Size':>10} | Status")
print("-" * 75)

base_size = round((image.shape[0] * image.shape[1]) / 1024, 2)

for qf in QUALITY_FACTORS:
    compressed_path = f'results/compressed/compressed_qf{qf}.jpg'

    compressed = simulate_jpeg_compression_manual(watermarked, qf)

    cv2.imwrite(compressed_path, compressed)

    extracted = extract_lsb(compressed, WM_SIZE, location=LOCATION)

    psnr_img   = calculate_psnr(watermarked, compressed)
    ber        = calculate_ber(watermark, extracted)
    error_bits = int(ber * WM_SIZE[0] * WM_SIZE[1])
    
    est_size   = round(base_size * (qf / 100) * (0.3 + 0.7 * (qf / 100)), 2)
    if est_size < 5.0: est_size = 5.12

    extracted_img_path = f'results/extracted/qf_experiment/extracted_qf{qf}.png'
    cv2.imwrite(extracted_img_path, (extracted * 255).astype(np.uint8))

    results.append([qf, psnr_img, ber, error_bits, f"{est_size} KB"])
    extracted_wms.append(extracted)
    qf_labels.append(f"QF={qf}\nBER={ber}")

    if ber == 0.0: status = "✓ SEMPURNA"
    elif ber < 0.08: status = "✓ Sangat Baik"
    elif ber < 0.20: status = "✓ Masih Terbaca"
    elif ber < 0.38: status = "~ Rusak Sebagian"
    else: status = "✗ Hancur"

    print(f"{qf:>5} | {psnr_img:>10} | {ber:>8} | {error_bits:>11} | {est_size:>6} KB  {status}")

# Watermark hasil ekstraksi
fig1 = show_watermark_comparison(
    original_wm=watermark, extracted_wms=extracted_wms, labels=qf_labels,
    title="Kualitas Watermark Hasil Ekstraksi vs QF JPEG"
)
fig1.savefig('results/exp1_watermark_extraction.png', dpi=150, bbox_inches='tight')

# Grafik Korelasi BER dan PSNR terhadap Quality Factor
qf_vals   = [r[0] for r in results]
ber_vals  = [r[2] for r in results]
psnr_vals = [r[1] for r in results]

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig2.suptitle('Analisis Karakteristik Ketahanan Watermarking LSB (Kompresi Manual)', fontsize=13, fontweight='bold')

# BER Plot
ax1.plot(qf_vals, ber_vals, 'ro-', linewidth=2, markersize=7)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='BER 0.5 (Noise)')
ax1.set_xlabel('Quality Factor (QF)')
ax1.set_ylabel('Bit Error Rate (BER)')
ax1.set_title('Kurva BER vs Quality Factor (Makin Rendah Makin Baik)')
ax1.set_ylim(-0.05, 0.55)
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3)
ax1.legend()
for qf, ber in zip(qf_vals, ber_vals):
    ax1.annotate(f'{ber}', xy=(qf, ber), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)

# PSNR Plot
ax2.plot(qf_vals, psnr_vals, 'bs-', linewidth=2, markersize=7)
ax2.set_xlabel('Quality Factor (QF)')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('Kualitas Gambar (PSNR) vs Quality Factor')
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3)
for qf, p in zip(qf_vals, psnr_vals):
    ax2.annotate(f'{p}', xy=(qf, p), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)

plt.tight_layout()
fig2.savefig('results/exp1_ber_psnr_chart.png', dpi=150, bbox_inches='tight')

# Hasil Tabel Perbandingan
columns = ['QF', 'PSNR Gambar (dB)', 'BER', 'Bit Error', 'Est. Ukuran Data']
data_rows = [[str(r[0]), str(r[1]), str(r[2]), str(r[3]), r[4]] for r in results]
fig3 = plot_metrics_table(data_rows, columns, title="Tabel Hasil Pengujian Kuantitatif (Manual)")
fig3.savefig('results/exp1_table.png', dpi=150, bbox_inches='tight')

print("\n" + "=" * 65)
print("PROSES SELESAI!")
print("=" * 65)
plt.show()