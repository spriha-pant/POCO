import pandas as pd
import numpy as np

# INPUT CSV
csv_file = "C_traces_cells_all_non_cells_excluded_csv.csv"

# OUTPUT NPZ
output_file = "barik_mouse_mousmi1.npz"

print("Loading CSV...")

df = pd.read_csv(csv_file)

# Remove the time column
activity = df.drop(columns=["time"]).values

print("Original shape (time, neurons):", activity.shape)

# Transpose to (neurons, time)
activity = activity.T

print("Transposed shape (neurons, time):", activity.shape)

# Save in NPZ format expected by POCO
np.savez(output_file, M=activity)

print("Saved:", output_file)