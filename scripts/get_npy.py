import numpy as np
import umap

dataset: dict[str, np.ndarray] = np.load(
        file = "./dataset.npz",
        mmap_mode = "r",
        allow_pickle = False,  
)
spks_big_array: np.ndarray     = dataset["spks"]

np.save(
    file = "spk_data",
    arr = spks_big_array,
    allow_pickle= False,
)
