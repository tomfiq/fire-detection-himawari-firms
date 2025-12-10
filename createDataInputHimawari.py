import numpy as np
from netCDF4 import Dataset
import os
import glob

# Tentukan direktori yang berisi file data satelit F:\satelitHimawari\dataKebakaran
data_dir = r'D:\penelitian\hibah 2024\data\data Satelit\netcdf'
files = glob.glob(os.path.join(data_dir, '**/*.nc'), recursive=True)

# Batasan koordinat untuk Region of Interest (ROI)
lat_lower = -3.650602881343665
lat_upper = 1.469398964128776
lon_lower = 110.73089588122669
lon_upper = 115.85088234776212

channels = ['tbb_07']
channel_folders = ['ch7']

root_dir = r'D:\penelitian\hibah 2024\data\data Satelit\dataSatelitNPZ'

for file_path in files:
    print("Memproses file:", file_path)
    data = Dataset(file_path)
    latitude = data.variables['latitude'][:]
    longitude = data.variables['longitude'][:]

    lat_indices = np.where((latitude >= lat_lower) & (latitude <= lat_upper))[0]
    lon_indices = np.where((longitude >= lon_lower) & (longitude <= lon_upper))[0]

    base_name = os.path.basename(file_path)
    date_time_str = base_name.split('_')[2] + '_' + base_name.split('_')[3]
    folder_path = os.path.join(root_dir, 'ch1')
    output_file_path = os.path.join(folder_path, f"{date_time_str}.npz")

    if os.path.exists(output_file_path):
        print(f"File {output_file_path} sudah ada untuk semua folder, lewati konversi.")
        continue

    for channel, folder_name in zip(channels, channel_folders):
        folder_path = os.path.join(root_dir, folder_name)
        output_file_path = os.path.join(folder_path, f"{date_time_str}.npz")
        channel_data = data.variables[channel][:]

        subset_data = channel_data[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]

        if isinstance(subset_data, np.ma.MaskedArray):
            subset_data = subset_data.data

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savez(output_file_path, channel_data=subset_data)

    data.close()

print("Ekstraksi dan penyimpanan data selesai.")
