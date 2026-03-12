# Cluster Summary Generation Prompt

## Context
You are analyzing clusters of artworks generated from an auction database. You will be provided with a JSON file `cluster_metadata_{label}.json` which contains:
- `cluster_id`: The ID of the cluster.
- `avg_price`: The mean sale price in USD.
- `median_price`: The median sale price in USD.
- `min_price`: The minimum sale price in USD.
- `max_price`: The maximum sale price in USD.
- `representative_artworks`: A list of 10 artworks closest to the cluster centroid, including their `title` and `artist`.

## Task
For each cluster in the JSON, generate a concise (single-sentence) summary that captures the "vibe" or "common theme" of the artworks in that cluster. 

## Instructions
1. Read the list of representative artworks carefully. Look for commonalities in artists, styles, or subjects.
2. Consider the price range. Is this a high-end "Masterpiece" cluster or a more accessible "Prints/Multiples" cluster?
3. Provide your output as a JSON mapping of `cluster_id` (as a string) to its corresponding `description`.

## Example Output Format
```json
{
  "0": "A collection of high-value contemporary abstracts and expressive works by major modern masters.",
  "1": "Mid-range decorative prints and lithographs featuring botanical and landscape themes.",
  ...
}
```

## Input Data
Please examine the `representative_artworks` and price stats in `02_embedding/models/cluster_metadata_image_only.json` and `02_embedding/models/cluster_metadata_multimodal.json` (once they are generated).
