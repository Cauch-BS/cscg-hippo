import numpy as np

dataset = np.load(
        file = "./dataset.npz",
        mmap_mode = "r",
        allow_pickle = False,  
)
spks_big_array: np.ndarray     = dataset["spks_big_array"]
day_ind_array: np.ndarray = dataset["day_ind_array"]

np.save(
    file = "spk_data",
    arr = spks_big_array,
    allow_pickle= False,
)

np.save(
    file = "day_ind_array",
    arr = day_ind_array,
    allow_pickle = False,
)
