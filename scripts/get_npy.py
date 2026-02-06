import numpy as np

dataset: dict[str, np.ndarray] = np.load(
        file = "./vr2p_extracted_signals.npz",
        mmap_mode = "r",
        allow_pickle = False,  
)
spks_big_array: np.ndarray     = dataset["spks"]
day_ind_array: np.ndarray      = dataset["day_ind"]

np.save(
    file = "spk_data",
    arr = spks_big_array,
    allow_pickle = False,
)

np.save(
    file = "day_data",
    arr = day_ind_array,
    allow_pickle = False
)
