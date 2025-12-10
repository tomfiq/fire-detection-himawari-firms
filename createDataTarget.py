import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

# =========================
# KONFIGURASI & INPUT FILES
# =========================
file_paths = [
    r'D:\penelitian\hibah 2025\data\fire_archive_J1V-C2_663864.csv',  # VIIRS NOAA-20 (C2)
    r'D:\penelitian\hibah 2025\data\fire_archive_M-C61_663863.csv',   # MODIS C6.1
    r'D:\penelitian\hibah 2025\data\fire_archive_SV-C2_663865.csv',   # VIIRS Suomi-NPP (C2)
]

output_directory = r'D:\penelitian\hibah 2025\data\target_npz'  # ganti sesuai kebutuhan

# ROI (Kalteng)
lat_lower = -3.650602881343665
lat_upper =  1.469398964128776
lon_lower = 110.73089588122669
lon_upper = 115.85088234776212
bbox = [lon_lower, lat_lower, lon_upper, lat_upper]

# Ambang kualitas/intensitas
CONF_THR_MODIS = 0
VIIRS_KEEP = {'n', 'h', 'nominal', 'high'}  # kategori yang diterima
BRIGHT_THR = 0
FRP_THR = 0

# Resolusi grid keluaran
RESOLUTION = (256, 256)  # (rows, cols)
CADENCE_MIN = 10         # step waktu 10 menit
HALF_WIN = 5            # ← jendela ±5 menit (KIRI–KANAN TERTUTUP)

# =========================
# UTILITAS
# =========================
def infer_source_from_name(path):
    name = os.path.basename(path).lower()
    if 'm-c6' in name or '_m-c61' in name:
        return 'MODIS'
    if 'sv-c2' in name:
        return 'VIIRS_NPP'      # Suomi-NPP
    if 'j1v-c2' in name:
        return 'VIIRS_NOAA20'   # NOAA-20
    return 'UNKNOWN'

def load_and_harmonize(path):
    """
    Kembalikan DF dengan kolom baku:
    latitude, longitude, frp, brightness, acq_date, acq_time, datetime, confidence_raw, source
    """
    df = pd.read_csv(path, low_memory=False)
    src = infer_source_from_name(path)

    # brightness: MODIS=brightness, VIIRS=bright_ti4
    if 'brightness' in df.columns:          # MODIS
        bright = df['brightness']
    elif 'bright_ti4' in df.columns:        # VIIRS 375m
        bright = df['bright_ti4']
    else:
        raise ValueError(f"Kolom brightness/bright_ti4 tidak ditemukan: {path}")

    required = ['latitude', 'longitude', 'frp', 'acq_date', 'acq_time', 'confidence']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang di {path}: {missing}")

    use = pd.DataFrame({
        'latitude':   pd.to_numeric(df['latitude'],  errors='coerce'),
        'longitude':  pd.to_numeric(df['longitude'], errors='coerce'),
        'frp':        pd.to_numeric(df['frp'],       errors='coerce'),
        'brightness': pd.to_numeric(bright,          errors='coerce'),
        'acq_date':   df['acq_date'].astype(str),
        'acq_time':   df['acq_time'].astype(str).str.zfill(4),
        'confidence_raw': df['confidence'],   # bisa angka/kata
        'source': src
    })

    # parse datetime (UTC)
    use['datetime'] = pd.to_datetime(
        use['acq_date'] + ' ' + use['acq_time'], format='%Y-%m-%d %H%M', errors='coerce'
    )

    # drop baris yang tidak valid
    use = use.dropna(subset=['latitude','longitude','frp','brightness','datetime']).reset_index(drop=True)
    return use

def confidence_ok(row):
    """
    MODIS: numeric >= CONF_THR_MODIS
    VIIRS: kategori 'nominal'/'high' (terima 'n','h','nominal','high')
    Jika bertipe numerik di VIIRS (kasus tertentu), fallback: >= CONF_THR_MODIS
    """
    src = row['source']
    raw = row['confidence_raw']

    # coba numerik dulu
    val = pd.to_numeric(raw, errors='coerce')
    if src == 'MODIS':
        return bool(val >= CONF_THR_MODIS) if pd.notna(val) else False

    # VIIRS C2: kategorikal
    s = str(raw).strip().lower()
    if s in VIIRS_KEEP or s[:1] in {'n','h'}:
        return True

    # fallback jika ternyata numerik
    if pd.notna(val):
        return bool(val >= CONF_THR_MODIS)
    return False

def floor_to(dt, minutes):
    return dt - timedelta(minutes=dt.minute % minutes,
                          seconds=dt.second, microseconds=dt.microsecond)

def ceil_to(dt, minutes):
    f = floor_to(dt, minutes)
    return f if dt == f else f + timedelta(minutes=minutes)

# =========================
# MUAT & GABUNGKAN SEMUA CSV
# =========================
all_dfs = []
for p in file_paths:
    print(f"Memuat: {p}")
    all_dfs.append(load_and_harmonize(p))

df = pd.concat(all_dfs, ignore_index=True)
print(f"Total baris gabungan sebelum filter: {len(df):,}")

# Buat flag confidence per baris
df['conf_ok'] = df.apply(confidence_ok, axis=1)

# =========================
# FILTER ROI + KUALITAS
# =========================
lon_min, lat_min, lon_max, lat_max = bbox
mask_roi = (
    (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
    (df['latitude']  >= lat_min) & (df['latitude']  <= lat_max)
)
mask_qc = (
    (df['brightness'] >= BRIGHT_THR) &
    (df['frp']        >= FRP_THR) &
    (df['conf_ok'])
)
df = df[mask_roi & mask_qc].reset_index(drop=True)
print(f"Total baris setelah filter ROI & mutu: {len(df):,}")
print(df['source'].value_counts())

# =========================
# SIAPKAN OUTPUT FOLDER
# =========================
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
os.makedirs(output_directory, exist_ok=True)

# =========================
# FUNGSI BANGUN MATRIKS
# =========================
def create_fire_matrix(start_time, end_time, df_all, output_file, resolution=RESOLUTION):
    # KIRI–KANAN TERTUTUP: [start_time, end_time]
    subset = df_all[(df_all['datetime'] >= start_time) & (df_all['datetime'] <= end_time)]
    if subset.empty:
        return  # event-only; hapus return bila ingin simpan frame kosong

    matrix = np.zeros(resolution, dtype=np.uint8)
    lon_res = (lon_max - lon_min) / resolution[1]
    lat_res = (lat_max - lat_min) / resolution[0]

    for _, row in subset.iterrows():
        c = int((row['longitude'] - lon_min) / lon_res)   # kolom
        r = int((row['latitude']  - lat_min) / lat_res)   # baris
        # Dilasi 3x3
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                rr, cc = r + di, c + dj
                if 0 <= rr < resolution[0] and 0 <= cc < resolution[1]:
                    matrix[rr, cc] = 1

    # simpan sebagai north-up (row 0 = utara)
    matrix = np.flipud(matrix)

    np.savez_compressed(
        output_file,
        matrix=matrix,
        metadata={
            'lon_min': lon_min, 'lat_min': lat_min,
            'lon_max': lon_max, 'lat_max': lat_max,
            'lon_res': lon_res, 'lat_res': lat_res,
            'window_policy': 'closed_closed',        # dokumentasi kebijakan jendela
            'half_window_min': HALF_WIN,
            'cadence_min': CADENCE_MIN,
            'roi': {'lat_lower': lat_lower, 'lat_upper': lat_upper,
                    'lon_lower': lon_lower, 'lon_upper': lon_upper}
        }
    )
    return len(subset)  # kembalikan jumlah event untuk logging

# =========================
# ITERASI 10-MENIT & SIMPAN NPZ
# =========================
# Ratakan ke grid 10-menit dengan floor/ceil
start_time = floor_to(df['datetime'].min(), CADENCE_MIN)
end_time   = ceil_to(df['datetime'].max(),  CADENCE_MIN)
print(f"Rentang waktu (diratakan ke grid): {start_time}  s.d.  {end_time}")

current_time = start_time
delta = timedelta(minutes=CADENCE_MIN)

# logging ringan: simpan jumlah event/ frame ke CSV agar bisa diaudit
log_rows = []

while current_time <= end_time:
    # jendela KIRI–KANAN TERTUTUP
    t0 = current_time - timedelta(minutes=HALF_WIN)
    t1 = current_time + timedelta(minutes=HALF_WIN)

    subset = df[(df['datetime'] >= t0) & (df['datetime'] <= t1)]
    n_ev = len(subset)
    if n_ev > 0:
        out_path = os.path.join(output_directory, f"{current_time.strftime('%Y%m%d_%H%M')}.npz")
        kept = create_fire_matrix(t0, t1, df, out_path, resolution=RESOLUTION) or 0

        by_src = subset['source'].value_counts().to_dict()
        log_rows.append({
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
            'n_events': int(n_ev),
            'n_modis': int(by_src.get('MODIS', 0)),
            'n_viirs_npp': int(by_src.get('VIIRS_NPP', 0)),
            'n_viirs_noaa20': int(by_src.get('VIIRS_NOAA20', 0)),
            't0': t0.strftime('%Y-%m-%d %H:%M'),
            't1': t1.strftime('%Y-%m-%d %H:%M'),
        })

    current_time += delta

if log_rows:
    log_df = pd.DataFrame(log_rows)
    log_csv = os.path.join(output_directory, 'target_event_log.csv')
    log_df.to_csv(log_csv, index=False)
    print(f"Log event per frame disimpan: {log_csv}")

print("Selesai membentuk target NPZ (event-only, window tertutup di kedua sisi, ±10 menit).")
