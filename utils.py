import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BIT_POSITION = 3  

def load_image(path, mode='gray'):
    if mode == 'gray':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    return img

def load_watermark(path, size):
    wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Watermark tidak ditemukan atau gagal dimuat: {path}")

    wm = cv2.resize(wm, size, interpolation=cv2.INTER_NEAREST)
    _, wm_binary = cv2.threshold(wm, 127, 1, cv2.THRESH_BINARY)
    return wm_binary

def embed_lsb(image, watermark_binary, location='top-left'):
    img = image.copy()
    wm_h, wm_w = watermark_binary.shape
    img_h, img_w = img.shape[:2]
    
    block_size = 2
    row_start, col_start = _get_location(location, img_h, img_w, wm_h * block_size, wm_w * block_size)
    
    mask = ~(1 << BIT_POSITION) & 0xFF

    for r in range(wm_h):
        for c in range(wm_w):
            bit = int(watermark_binary[r, c])
            
            r_img = row_start + (r * block_size)
            c_img = col_start + (c * block_size)
            
            if r_img + block_size <= img_h and c_img + block_size <= img_w:
                if len(img.shape) == 3:
                    region = img[r_img:r_img+block_size, c_img:c_img+block_size, 0]
                    img[r_img:r_img+block_size, c_img:c_img+block_size, 0] = (region & mask) | (bit << BIT_POSITION)
                else:
                    region = img[r_img:r_img+block_size, c_img:c_img+block_size]
                    img[r_img:r_img+block_size, c_img:c_img+block_size] = (region & mask) | (bit << BIT_POSITION)
                    
    return img

def extract_lsb(watermarked_image, wm_size, location='top-left'):
    img = watermarked_image.copy()
    wm_w, wm_h = wm_size
    img_h, img_w = img.shape[:2]
    
    block_size = 2
    row_start, col_start = _get_location(location, img_h, img_w, wm_h * block_size, wm_w * block_size)
    
    extracted = np.zeros((wm_h, wm_w), dtype=np.uint8)

    for r in range(wm_h):
        for c in range(wm_w):
            r_img = row_start + (r * block_size)
            c_img = col_start + (c * block_size)
            
            if r_img + block_size <= img_h and c_img + block_size <= img_w:
                if len(img.shape) == 3:
                    region = img[r_img:r_img+block_size, c_img:c_img+block_size, 0]
                else:
                    region = img[r_img:r_img+block_size, c_img:c_img+block_size]
                
                bits_in_block = (region >> BIT_POSITION) & 1
                
                if np.mean(bits_in_block) >= 0.5:
                    extracted[r, c] = 1
                else:
                    extracted[r, c] = 0
                    
    return extracted

def _get_location(location, img_h, img_w, wm_h, wm_w):
    if location == 'top-left': return 0, 0
    elif location == 'center': return (img_h - wm_h) // 2, (img_w - wm_w) // 2
    elif location == 'bottom-right': return img_h - wm_h, img_w - wm_w
    elif location == 'top-right': return 0, img_w - wm_w
    elif location == 'bottom-left': return img_h - wm_h, 0
    return 0, 0

def calculate_psnr(original, compressed):
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0: return float('inf')
    return round(20 * np.log10(255.0 / np.sqrt(mse)), 2)

def calculate_ber(original_wm, extracted_wm):
    min_h = min(original_wm.shape[0], extracted_wm.shape[0])
    min_w = min(original_wm.shape[1], extracted_wm.shape[1])
    orig = original_wm[:min_h, :min_w]
    extr = extracted_wm[:min_h, :min_w]
    return round(np.sum(orig != extr) / orig.size, 4)

def show_watermark_comparison(original_wm, extracted_wms, labels, title="Perbandingan Watermark"):
    n = len(extracted_wms) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.5))
   
    axes[0].imshow(original_wm * 255, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original\nWatermark", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for i, (wm, label) in enumerate(zip(extracted_wms, labels)):
        axes[i + 1].imshow(wm * 255, cmap='gray', vmin=0, vmax=255)
        axes[i + 1].set_title(label, fontsize=9)
        axes[i + 1].axis('off')

    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig

def plot_metrics_table(data_rows, columns, title="Tabel Evaluasi"):
    fig, ax = plt.subplots(figsize=(len(columns) * 1.8, len(data_rows) * 0.5 + 1.0))
    ax.axis('off')
    table = ax.table(cellText=data_rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(len(columns)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig