"""
Local smoke test — bypasses RunPod, calls handler() directly.

Usage:
    uv run python test_local.py

Generates synthetic embeddings so no real data is needed.
Requires ~2GB RAM for LaBSE model load.
"""

import json
import numpy as np
from src.handler import handler

# Generate synthetic data: 30 items with random 768-dim embeddings
np.random.seed(42)
N = 30

items = []
sample_texts = [
    "The professor explains the lessons very clearly and patiently",
    "Magaling magturo ang teacher namin sa subject na ito",
    "Maayo kaayo siya mo explain sa among lessons",
    "The teaching method is effective and engaging for students",
    "Hindi ko maintindihan ang mga lessons dahil mabilis magturo",
    "Nindot ang pagtudlo sa among propesor labi na sa math",
    "I wish the professor would give more examples during class",
    "Kulang ang materials na ibinibigay sa amin para sa review",
    "The professor is always late and dismisses class early",
    "Very helpful and approachable when we have questions about topics",
]

for i in range(N):
    text = sample_texts[i % len(sample_texts)]
    # Create clustered embeddings (so BERTopic can find topics)
    cluster = i % 3
    base = np.random.randn(768).astype(np.float32)
    base += cluster * 2  # shift clusters apart
    base = base / np.linalg.norm(base)  # normalize

    items.append({
        "submissionId": f"sub-{i:03d}",
        "text": text,
        "embedding": base.tolist(),
    })

event = {
    "input": {
        "jobId": "test-local-001",
        "version": "1.0",
        "type": "topic-model",
        "items": items,
        "params": {
            "min_topic_size": 5,
            "nr_topics": 3,
            "umap_n_neighbors": 10,
            "umap_n_components": 5,
        },
        "metadata": {"pipelineId": "p-local", "runId": "r-local"},
        "publishedAt": "2026-03-16T00:00:00.000Z",
    }
}

print(f"Sending {N} items to handler...")
result = handler(event)

print(f"\nStatus: {result['status']}")
if result["status"] == "completed":
    print(f"Topics: {len(result['topics'])}")
    for t in result["topics"]:
        print(f"  Topic {t['topicIndex']}: {t['rawLabel']} ({t['docCount']} docs)")
        print(f"    Keywords: {', '.join(t['keywords'][:5])}")
    print(f"Assignments: {len(result['assignments'])}")
    print(f"Outliers: {result['outlierCount']}")
    print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
else:
    print(f"Error: {result.get('error')}")
