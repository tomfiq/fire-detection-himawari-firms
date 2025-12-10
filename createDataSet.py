import os
import re
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

# ======================================================
# Util baca NPZ
# ======================================================
def load_npz_key(path: str, key: str):
    with np.load(path) as d:
        return d[key]

def get_timestamps_from_channel_dir(channel_dir: str) -> List[str]:
    """Ambil semua timestamp valid (YYYYMMDD_HHMM.npz) dan urutkan."""
    if not os.path.isdir(channel_dir):
        raise FileNotFoundError(f"Channel dir tidak ditemukan: {channel_dir}")
    patt = re.compile(r'\d{8}_\d{4}\.npz$')
    files = [f for f in os.listdir(channel_dir) if patt.match(f)]
    return sorted([f[:-4] for f in files])  # tanpa ".npz"

def ensure_pairs(input_dir: str, target_dir: str, channel_indices: List[int], target_key='matrix') -> List[str]:
    """Hanya timestamp yang memiliki SEMUA channel & target."""
    base_ch_dir = os.path.join(input_dir, f'ch{channel_indices[0]}')
    ts_all = get_timestamps_from_channel_dir(base_ch_dir)
    ok = []
    for ts in ts_all:
        chans_ok = all(os.path.isfile(os.path.join(input_dir, f'ch{ch}', f'{ts}.npz')) for ch in channel_indices)
        tgt_ok   = os.path.isfile(os.path.join(target_dir, f'{ts}.npz'))
        if chans_ok and tgt_ok:
            ok.append(ts)
    return ok

# ======================================================
# Hitung jumlah titik api per timestamp
# ======================================================
def count_fire_points_for_ts(target_dir: str, timestamps: List[str], target_key='matrix', threshold=None) -> np.ndarray:
    """
    Mengembalikan array counts sepanjang timestamps.
    - Jika target biner 0/1: biarkan threshold=None (hitung nonzero).
    - Jika target kontinu: set threshold (contoh >0 atau FRP>=20).
    """
    counts = []
    for ts in tqdm(timestamps, desc="Counting fire points"):
        m = load_npz_key(os.path.join(target_dir, f'{ts}.npz'), target_key)
        if threshold is None:
            cnt = np.count_nonzero(m)
        else:
            cnt = np.count_nonzero(m > threshold)
        counts.append(int(cnt))
    return np.array(counts, dtype=np.int64)

# ======================================================
# Binning kuantil (robust)
# ======================================================
def make_bins_from_counts(counts: np.ndarray, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kembalikan (bin_ids, bin_edges).
    - Jika nilai unik sangat sedikit, fallback ke bin berdasarkan nilai unik.
    - Kuantil memberi sebaran merata per bin.
    """
    uniq = np.unique(counts)
    if len(uniq) <= 1:
        return np.zeros_like(counts, dtype=np.int32), np.array([counts.min(), counts.max()])

    if len(uniq) <= n_bins:
        edges = np.r_[uniq, [uniq[-1] + 1]]  # half-open
        bin_ids = np.digitize(counts, edges[1:-1], right=False)
        return bin_ids.astype(np.int32), edges

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(counts, qs)
    edges = np.unique(edges)
    if len(edges) == 1:
        return np.zeros_like(counts, dtype=np.int32), np.array([edges[0], edges[0]])

    eps = 1e-9
    edges[0] -= eps
    edges[-1] += eps
    bin_ids = np.digitize(counts, edges[1:-1], right=False)
    return bin_ids.astype(np.int32), edges

# ======================================================
# Stratified split per bin (dengan opsi group-by-date)
# ======================================================
def stratified_split_by_bins(
    timestamps: List[str],
    bin_ids: np.ndarray,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    group_by_date: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    timestamps = np.array(timestamps)
    bin_ids = np.array(bin_ids)

    train, val, test = [], [], []

    if not group_by_date:
        # Sederhana: acak di dalam tiap bin sesuai rasio target
        for b in np.unique(bin_ids):
            idx = np.where(bin_ids == b)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_tr = int(round(ratios[0]*n))
            n_va = int(round(ratios[1]*n))
            n_te = n - n_tr - n_va
            train.extend(timestamps[idx[:n_tr]])
            val.extend(timestamps[idx[n_tr:n_tr+n_va]])
            test.extend(timestamps[idx[n_tr+n_va:]])
        return sorted(train), sorted(val), sorted(test)

    # --- Versi group_by_date (tiap tanggal tidak menyebar beberapa split) ---
    def date_of(ts: str) -> str:
        return ts.split('_')[0]  # 'YYYYMMDD'

    for b in np.unique(bin_ids):
        idx_bin = np.where(bin_ids == b)[0]
        n = len(idx_bin)
        n_tr = int(round(ratios[0]*n))
        n_va = int(round(ratios[1]*n))
        n_te = n - n_tr - n_va

        cur_tr = cur_va = cur_te = 0

        # kelompokkan indeks bin ini per tanggal
        by_date: Dict[str, List[int]] = {}
        for i in idx_bin:
            d = date_of(timestamps[i])
            by_date.setdefault(d, []).append(i)

        # urutkan grup terbesar dulu agar packing efisien
        groups = sorted(by_date.items(), key=lambda kv: -len(kv[1]))

        for d, idx_list in groups:
            size = len(idx_list)
            rem_tr = n_tr - cur_tr
            rem_va = n_va - cur_va
            rem_te = n_te - cur_te

            # jika masih ada defisit, isi yang defisit terbesar
            if rem_tr > 0 or rem_va > 0 or rem_te > 0:
                options = [('tr', rem_tr), ('va', rem_va), ('te', rem_te)]
                options.sort(key=lambda x: x[1], reverse=True)
                choice = options[0][0]
            else:
                # semua sudah memenuhi target -> pilih overfill minimum
                over_tr = abs((cur_tr + size) - n_tr)
                over_va = abs((cur_va + size) - n_va)
                over_te = abs((cur_te + size) - n_te)
                choices = [('tr', over_tr), ('va', over_va), ('te', over_te)]
                # tie-break: train > val > test
                choices.sort(key=lambda x: (x[1], {'tr':0, 'va':1, 'te':2}[x[0]]))
                choice = choices[0][0]

            if choice == 'tr':
                train.extend(timestamps[idx_list]); cur_tr += size
            elif choice == 'va':
                val.extend(timestamps[idx_list]);   cur_va += size
            else:
                test.extend(timestamps[idx_list]);  cur_te += size

    return sorted(train), sorted(val), sorted(test)

def stratified_time_fire_splits(
    input_dir: str,
    target_dir: str,
    channel_indices: List[int],
    target_key='matrix',
    threshold=None,
    n_bins=5,
    ratios=(0.7, 0.15, 0.15),
    seed=42,
    group_by_date=True
):
    ts_all = ensure_pairs(input_dir, target_dir, channel_indices, target_key=target_key)
    if len(ts_all) == 0:
        raise RuntimeError("Tidak ada pasangan input-target yang valid.")

    counts = count_fire_points_for_ts(target_dir, ts_all, target_key=target_key, threshold=threshold)
    bin_ids, edges = make_bins_from_counts(counts, n_bins=n_bins)

    tr, va, te = stratified_split_by_bins(
        ts_all, bin_ids, ratios=ratios, seed=seed, group_by_date=group_by_date
    )

    # ringkasan distribusi per bin
    def hist_for(ts_list):
        idx_map = {t:i for i,t in enumerate(ts_all)}
        idx = np.array([idx_map[t] for t in ts_list], dtype=int)
        return np.bincount(bin_ids[idx], minlength=len(np.unique(bin_ids)))

    summary = {
        "total": len(ts_all),
        "bins_edges": edges.tolist(),
        "train": {"n": len(tr), "bin_hist": hist_for(tr).tolist()},
        "val":   {"n": len(va), "bin_hist": hist_for(va).tolist()},
        "test":  {"n": len(te), "bin_hist": hist_for(te).tolist()},
    }
    return tr, va, te, ts_all, counts, bin_ids, summary

# ======================================================
# Bangun array (X/Y) & simpan ke .npz untuk tiap split
# ======================================================
def load_and_combine_channels(base_dir: str, timestamp: str, channel_indices: List[int], input_key='channel_data'):
    """Stack channel di axis terakhir -> (H, W, C)."""
    arrs = []
    for ch in channel_indices:
        fp = os.path.join(base_dir, f'ch{ch}', f'{timestamp}.npz')
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"File channel hilang: {fp}")
        arrs.append(load_npz_key(fp, input_key))
    return np.stack(arrs, axis=-1).astype(np.float32)

def build_arrays_for_split(
    split_timestamps: List[str],
    input_dir: str,
    target_dir: str,
    channel_indices: List[int],
    input_key='channel_data',
    target_key='matrix',
    desc: str = "Loading"
):
    inputs, targets, names = [], [], []
    for ts in tqdm(split_timestamps, desc=desc):
        x = load_and_combine_channels(input_dir, ts, channel_indices, input_key=input_key)   # (H, W, C)
        y = load_npz_key(os.path.join(target_dir, f'{ts}.npz'), target_key).astype(np.float32)  # (H, W) atau (H, W, K)
        inputs.append(x)
        targets.append(y)
        names.append(ts)
    X = np.stack(inputs, axis=0)   # (N, H, W, C)
    Y = np.stack(targets, axis=0)  # (N, ...)
    names = np.array(names)
    return X, Y, names

def save_npz_split(save_dir: str, base_name: str, split_name: str, X: np.ndarray, Y: np.ndarray, names: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    out_fp = os.path.join(save_dir, f"{base_name}_{split_name}.npz")
    np.savez(out_fp, input_np=X, target_np=Y, file_names=names)
    print(f"[OK] Tersimpan: {out_fp} | X{X.shape} Y{Y.shape} N={len(names)}")

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    # ==== PARAMETER YANG PERLU DISesuaikan ====
    input_dir   = r"D:/penelitian/hibah 2024/data/data Satelit/dataKebakaran"
    target_dir  = r"D:/penelitian/hibah 2024/data/data Satelit/dataKebakaran/target"
    channel_indices = [7]            # contoh: hanya ch7
    input_key   = "channel_data"     # kunci NPZ input
    target_key  = "matrix"           # kunci NPZ target
    threshold   = None               # None untuk biner; mis. 0 atau 20 untuk kontinu
    n_bins      = 5
    ratios      = (0.70, 0.15, 0.15) # train/val/test
    seed        = 42
    group_by_date = True             # sarankan True agar 1 tanggal tidak menyebar

    dataset_name = "datasetIR(7)_buffer10m_Nofilter"
    save_dir     = r"D:/penelitian/hibah 2024/data/data Satelit/dataKebakaran4"

    # ==== STRATIFIED SPLIT ====
    tr, va, te, ts_all, counts, bin_ids, summary = stratified_time_fire_splits(
        input_dir, target_dir, channel_indices,
        target_key=target_key, threshold=threshold,
        n_bins=n_bins, ratios=ratios, seed=seed, group_by_date=group_by_date
    )

    print("Ringkasan stratifikasi (bin berdasarkan jumlah titik api):")
    print(summary)

    def safe_range(name, arr):
        if len(arr) == 0:
            return f"{name}: (kosong)"
        return f"{name}: {arr[0]} \u2192 {arr[-1]}"

    print("Contoh range tanggal:")
    print(" ", safe_range("Train", tr))
    print(" ", safe_range("Val  ", va))
    print(" ", safe_range("Test ", te))

    # ==== BANGUN ARRAY & SIMPAN ====
    if len(tr):
        X_tr, Y_tr, N_tr = build_arrays_for_split(tr, input_dir, target_dir, channel_indices,
                                                  input_key=input_key, target_key=target_key, desc="Loading train")
        save_npz_split(save_dir, dataset_name, "train", X_tr, Y_tr, N_tr)
    if len(va):
        X_va, Y_va, N_va = build_arrays_for_split(va, input_dir, target_dir, channel_indices,
                                                  input_key=input_key, target_key=target_key, desc="Loading val")
        save_npz_split(save_dir, dataset_name, "val", X_va, Y_va, N_va)
    if len(te):
        X_te, Y_te, N_te = build_arrays_for_split(te, input_dir, target_dir, channel_indices,
                                                  input_key=input_key, target_key=target_key, desc="Loading test")
        save_npz_split(save_dir, dataset_name, "test", X_te, Y_te, N_te)

    print("Selesai membuat & menyimpan tiga split .npz (train/val/test).")

    # ==== OPSIONAL: sanity-check histogram kembali ====
    def check_hist(ts_list, ts_all, bin_ids):
        idx_map = {t:i for i,t in enumerate(ts_all)}
        idx = np.array([idx_map[t] for t in ts_list], dtype=int)
        return np.bincount(bin_ids[idx], minlength=len(np.unique(bin_ids)))

    if len(tr) and len(va) and len(te):
        print("Bin hist train:", check_hist(tr, ts_all, bin_ids))
        print("Bin hist val  :", check_hist(va, ts_all, bin_ids))
        print("Bin hist test :", check_hist(te, ts_all, bin_ids))
