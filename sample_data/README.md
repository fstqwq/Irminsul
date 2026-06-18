# ICPC World Finals Retrieval Sample

The sample keeps only ICPC World Finals problems from QOJ category 11. It is intended for retrieval experiments where each query should retrieve the problem with the same `id`.

## Files

- `documents.jsonl`: 196 ICPC World Finals problem statements.
- `queries.jsonl`: 196 generated natural-language queries for the same problems, produced through OpenRouter with several models.

The two files have the same `id` order. Due to crawler issues and intentionally varied query quality, not every row is guaranteed to be correct. The current method has 192/196 Hit@1 when evaluating with top-10 embedding results.

## Schema

`documents.jsonl` rows:

```json
{
  "id": "QOJ/10463",
  "title": "To Add or to Multiply",
  "text": "...",
  "url": "https://qoj.ac/problem/10463",
  "calibrate_sources": []
}
```

`queries.jsonl` rows:

```json
{
  "id": "QOJ/10463",
  "title": "To Add or to Multiply",
  "url": "https://qoj.ac/problem/10463",
  "query": "...",
  "source": {
    "provider": "openrouter",
    "model": "...",
    "provider_name": "..."
  }
}
```
