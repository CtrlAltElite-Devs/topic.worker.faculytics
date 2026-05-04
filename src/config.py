"""
Configuration for the topic modeling worker.
"""

import torch

# Model
LABSE_MODEL = "sentence-transformers/LaBSE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Worker version (returned in responses)
WORKER_VERSION = "1.1.0"

# RUN 012 defaults — proven optimal params from experimentation
DEFAULT_PARAMS = {
    "min_topic_size": 15,
    "nr_topics": 20,
    "umap_n_neighbors": 20,
    "umap_n_components": 10,
}
