# TED Talks Recommender — Evidence-based EDA Findings

- Dataset size: **2,550 talks** and **20 columns**.
- Transcript coverage: **96.6%**.
- Description coverage: **100.0%**.
- Tag coverage: **100.0%**.
- Views coverage: **100.0%**.

## Recommended implementation direction
1. Start with transcript-based content similarity as the primary baseline.
2. Use tags as interpretable topic filters and as an additional matching signal.
3. Use popularity (views) only as a secondary ranking feature; relevance must remain central.
4. Add duration and speaker filters only when those fields are available and validated.
5. Do not claim collaborative filtering from this dataset alone: no clear user-level interaction fields were detected.

## Recommended pipeline
User selects topics / enters an interest
        ↓
Build a user-interest representation from selected topics or liked talks
        ↓
Generate candidate talks using TF-IDF or sentence embeddings
        ↓
Re-rank using topic match + semantic relevance + optional popularity/duration
        ↓
Show recommendations with explanations (matched topics / similar content)

## Important limitation
This dataset-level EDA cannot prove recommendation quality. Module 3 should build text vectors; Module 4 should implement and evaluate recommendations with suitable relevance labels or user feedback.