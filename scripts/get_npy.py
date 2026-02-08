import numpy as np

<<<<<<< HEAD
dataset: dict[str, np.ndarray] = np.load(
        file = "./vr2p_extracted_signals.npz",
        mmap_mode = "r",
        allow_pickle = False,  
)
spks_big_array: np.ndarray     = dataset["spks"]
day_ind_array: np.ndarray      = dataset["day_ind"]
=======
dataset = np.load(
        file = "./dataset.npz",
        mmap_mode = "r",
        allow_pickle = False,  
)
spks_big_array: np.ndarray     = dataset["spks_big_array"]
day_ind_array: np.ndarray = dataset["day_ind_array"]
>>>>>>> 601499a0d847e1a74dd3d4d49118f9c193aec64e

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

np.save(
    file = "day_ind_array",
    arr = day_ind_array,
    allow_pickle = False,
)
