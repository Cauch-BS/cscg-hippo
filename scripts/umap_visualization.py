import numpy as np
import umap

spks_big_array = np.load(
        file = "./spk_data.npy",
        mmap_mode = "r",
        allow_pickle = False,  
)
seed: int = 42

umap_data= umap.UMAP(
    n_neighbors = 100,
    n_components = 3,
    min_dist = 0.1,
    metric = 'correlation',
    random_state = seed,
    verbose = True, 
).fit(
    X = spks_big_array.T
)

umap_embedding = umap_data.embedding_

np.save(f'./embedding_{seed}.npy', umap_embedding)


