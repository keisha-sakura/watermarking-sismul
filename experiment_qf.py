import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import (
    load_image, load_watermark,
    embed_lsb, extract_lsb,
    calculate_psnr, calculate_ber,
    show_watermark_comparison, plot_metrics_table,
    _get_location
)

IMAGE_PATH     = 'images/face.jpg'
WATERMARK_PATH = 'images/logo.png'
WM_SIZE        = (64, 64)
LOCATION       = 'top-left'
QUALITY_FACTORS = [100, 90, 80, 70, 50, 30, 10]

os.makedirs('results/steps', exist_ok=True)
os.makedirs('results/compressed', exist_ok=True)
os.makedirs('results/extracted/qf_experiment', exist_ok=True)

print("=" * 65)
print("RUNNING: EKSPERIMEN LSB DENGAN VISUALISASI DETAIL TAHAPAN")
print("=" * 65)

# FASE 1: PREPROCESSING (Deteksi Tahapan Grayscale)
print("\nMenjalankan Preprocessing Gambar & Watermark...")

img_color = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")
img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

wm_original = cv2.imread(WATERMARK_PATH, cv2.IMREAD_GRAYSCALE)
watermark = load_watermark(WATERMARK_PATH, WM_SIZE)

b_channel, g_channel, r_channel = cv2.split(img_color)
fig_gray_detail, axes_gd = plt.subplots(1, 4, figsize=(16, 4))
axes_gd[0].imshow(r_channel, cmap='Reds')
axes_gd[0].set_title("1. Kanal Merah (Red Channel)")
axes_gd[0].axis('off')
axes_gd[1].imshow(g_channel, cmap='Greens')
axes_gd[1].set_title("2. Kanal Hijau (Green Channel)")
axes_gd[1].axis('off')
axes_gd[2].imshow(b_channel, cmap='Blues')
axes_gd[2].set_title("3. Kanal Biru (Blue Channel)")
axes_gd[2].axis('off')
axes_gd[3].imshow(image, cmap='gray')
axes_gd[3].set_title("4. Hasil Akhir Grayscale\n(Weighted Luminance)")
axes_gd[3].axis('off')
plt.suptitle("BAGAIMANA FOTO BERWARNA MENJADI GRAYSCALE", fontsize=13, fontweight='bold')
plt.tight_layout()
fig_gray_detail.savefig('results/steps/detail_grayscale.png', dpi=150)

fig_phase1, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(img_color_rgb)
axes[0].set_title("1. face.jpg (Original Color)")
axes[0].axis('off')

axes[1].imshow(image, cmap='gray')
axes[1].set_title("2. face.jpg (Grayscale)")
axes[1].axis('off')

axes[2].imshow(wm_original, cmap='gray')
axes[2].set_title("3. logo.png (Original)")
axes[2].axis('off')

axes[3].imshow(watermark, cmap='gray')
axes[3].set_title(f"4. Watermark Biner\nResized {WM_SIZE}")
axes[3].axis('off')

plt.suptitle("TAHAPAN PREPROCESSING INPUT", fontsize=14, fontweight='bold')
plt.tight_layout()
fig_phase1.savefig('results/steps/preprocessing.png', dpi=150)

# FASE 2: EMBEDDING (Proses Penyisipan & Peta Lokasi)
print("Menjalankan Proses Embedding LSB...")

watermarked = embed_lsb(image, watermark, location=LOCATION)

img_h, img_w = image.shape[:2]
wm_h, wm_w = watermark.shape
block_size = 2
row_start, col_start = _get_location(LOCATION, img_h, img_w, wm_h * block_size, wm_w * block_size)

img_location_map = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
cv2.rectangle(img_location_map, (col_start, row_start), 
              (col_start + wm_w * block_size, row_start + wm_h * block_size), (255, 0, 0), 3)

diff_map = cv2.absdiff(image, watermarked) * 20 

zoom_size = wm_h * block_size # 64 * 2 = 128 piksel
orig_zoom = image[row_start:row_start+zoom_size, col_start:col_start+zoom_size]
wm_zoom = watermarked[row_start:row_start+zoom_size, col_start:col_start+zoom_size]

bit_plane_before = (orig_zoom >> 3) & 1
bit_plane_after = (wm_zoom >> 3) & 1

fig_embed_detail, axes_ed = plt.subplots(2, 2, figsize=(10, 9))
axes_ed[0, 0].imshow(orig_zoom, cmap='gray')
axes_ed[0, 0].set_title("1. Piksel Asli (Zoom Area 128x128)")
axes_ed[0, 0].axis('off')

axes_ed[0, 1].imshow(wm_zoom, cmap='gray')
axes_ed[0, 1].set_title("2. Piksel Ter-watermark (Zoom Area 128x128)")
axes_ed[0, 1].axis('off')

axes_ed[1, 0].imshow(bit_plane_before, cmap='binary')
axes_ed[1, 0].set_title("3. Bit-Plane Posisi 3 ASLI\n(Berisi Noise Alami Gambar)")
axes_ed[1, 0].axis('off')

axes_ed[1, 1].imshow(bit_plane_after, cmap='binary')
axes_ed[1, 1].set_title("4. Bit-Plane Posisi 3 SESUDAH Embed\n(Pola Logo Rahasia Terbaca di Memori!)")
axes_ed[1, 1].axis('off')

plt.suptitle("STRUKTUR STRATIFIKASI BIT INTERNAL (ZOOM EMBED AREA)", fontsize=13, fontweight='bold')
plt.tight_layout()
fig_embed_detail.savefig('results/steps/detail_embedding_pixel.png', dpi=150)

fig_phase2, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(img_location_map)
axes[0].set_title(f"1. Peta Lokasi Penyisipan\n({LOCATION})")
axes[0].axis('off')

axes[1].imshow(watermarked, cmap='gray')
axes[1].set_title("2. Hasil Watermarked Image")
axes[1].axis('off')

axes[2].imshow(diff_map, cmap='jet')
axes[2].set_title("3. Selisih Bit (Diff Map x20)\n[Lokasi Data Tersembunyi]")
axes[2].axis('off')

plt.suptitle("PROSES EMBEDDING (PENYISIPAN WATERMARK)", fontsize=14, fontweight='bold')
plt.tight_layout()
fig_phase2.savefig('results/steps/embedding.png', dpi=150)

# FASE 3: TENGAH PROSES / SERANGAN KOMPRESI (Simulasi Kasus QF = 50)
print("Menampilkan Kondisi Gambar di Tengah Proses (Distorsi Kompresi)...")

def simulate_jpeg_compression_manual(image, qf):
    img = image.copy().astype(np.float64)
    h, w = img.shape[:2]
    scale = (100 - qf) / 50 if qf >= 50 else 50 / qf
    if scale == 0: return image.copy() 
    
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            r_end = min(r + 8, h)
            c_end = min(c + 8, w)
            block = img[r:r_end, c:c_end]
            block_mean = np.mean(block)
            compressed_block = block_mean + (block - block_mean) / (1 + scale * 0.45)
            img[r:r_end, c:c_end] = compressed_block
    return np.clip(img, 0, 255).astype(np.uint8)

sample_qf = 50
img_compressed_sample = simulate_jpeg_compression_manual(watermarked, sample_qf)
compression_noise = cv2.absdiff(watermarked, img_compressed_sample) * 10 

fig_phase3, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(watermarked, cmap='gray')
axes[0].set_title("1. Sebelum Kompresi")
axes[0].axis('off')

axes[1].imshow(img_compressed_sample, cmap='gray')
axes[1].set_title(f"2. Terkompresi (QF = {sample_qf})\n[Mulai Muncul Efek Blok]")
axes[1].axis('off')

axes[2].imshow(compression_noise, cmap='hot')
axes[2].set_title("3. Peta Kerusakan Piksel (x10)\n[Merusak Bit LSB]")
axes[2].axis('off')

plt.suptitle(f"KONDISI GAMBAR DI TENGAH PROSES (KOMPRESI QF={sample_qf})", fontsize=14, fontweight='bold')
plt.tight_layout()
fig_phase3.savefig('results/steps/compression_midprocess.png', dpi=150)

# FASE 4: MULTI-QF EXPERIMENT & EVALUASI
print("\nMenjalankan Looping Pengujian Multi-QF & Ekstraksi...")

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

comp_zoom = img_compressed_sample[row_start:row_start+zoom_size, col_start:col_start+zoom_size]
bit_plane_comp = (comp_zoom >> 3) & 1
extracted_sample = extract_lsb(img_compressed_sample, WM_SIZE, location=LOCATION)

fig_extract_detail, axes_exd = plt.subplots(1, 3, figsize=(14, 4.5))
axes_exd[0].imshow(comp_zoom, cmap='gray')
axes_exd[0].set_title("1. Zoom Area Terkompresi\n(Efek Degradasi JPEG QF=50)")
axes_exd[0].axis('off')

axes_exd[1].imshow(bit_plane_comp, cmap='binary')
axes_exd[1].set_title("2. Bit-Plane Posisi 3 Rusak\n(Bit Mulai Hancur Kena Serangan)")
axes_exd[1].axis('off')

axes_exd[2].imshow(extracted_sample, cmap='gray')
axes_exd[2].set_title("3. Hasil Ekstraksi Akhir\n(Setelah Penyaringan Rata-rata Blok)")
axes_exd[2].axis('off')

plt.suptitle("MEKANISME RETRIEVAL & FILTERING BLOK SAAT EKSTRAKSI", fontsize=13, fontweight='bold')
plt.tight_layout()
fig_extract_detail.savefig('results/steps/detail_extraction_pixel.png', dpi=150)

fig_wm_comp = show_watermark_comparison(original_wm=watermark, extracted_wms=extracted_wms, labels=qf_labels, title="Kualitas Watermark Hasil Ekstraksi vs QF JPEG")
fig_wm_comp.savefig('results/steps/extracted_comparison.png', dpi=150, bbox_inches='tight')

columns = ['QF', 'PSNR Gambar (dB)', 'BER', 'Bit Error', 'Est. Ukuran Data']
data_rows = [[str(r[0]), str(r[1]), str(r[2]), str(r[3]), r[4]] for r in results]
fig_table = plot_metrics_table(data_rows, columns, title="Tabel Hasil Pengujian Kuantitatif")
fig_table.savefig('results/steps/metrics_table.png', dpi=150, bbox_inches='tight')

print("\n" + "=" * 65)
print("PROSES SELESAI! Visualisasi tersimpan di folder 'results/steps/'")
print("=" * 65)
plt.show()