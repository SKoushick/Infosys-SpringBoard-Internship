# Module 1: Raw Dataset Validation Report

- **Generated At (UTC):** `2026-10-20 18:34:11 UTC`
- **Dataset File:** `Scraped/Data/Raw/ted_talks.csv` (`30.11 MB`)
- **Data Source & License:** TED Talks Public Archive (`ted_main.csv` + `transcripts.csv`), Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (`CC BY-NC-SA 4.0`)
- **Total Records:** `2550`
- **Total Columns:** `15`

## 1. Integrity & Quality Checks
- **Duplicate Rows:** `0`
- **Duplicate `talk_id`s:** `0`
- **Duplicate URLs:** `0`
- **Empty/Invalid Titles:** `0`
- **Valid TED Talk URLs:** `2550 / 2550` (`0` invalid)
- **Transcript Availability:** `2464 / 2550` (`96.63%` present, `86` missing)
- **Tag Availability:** `2550 / 2550` (`416` unique topic tags)
- **Top 10 Tags:** `technology` (727), `science` (567), `global issues` (501), `culture` (486), `tedx` (450), `design` (418), `business` (348), `entertainment` (299), `health` (236), `innovation` (229)

## 2. Column Schema & Missing Value Breakdown
| Column Name | Data Type | Missing / Empty Count | Missing % |
| :--- | :--- | :---: | :---: |
| `talk_id` | `int64` | 0 | 0.00% |
| `title` | `str` | 0 | 0.00% |
| `speaker` | `str` | 0 | 0.00% |
| `speaker_occupation` | `str` | 6 | 0.24% |
| `description` | `str` | 0 | 0.00% |
| `tags` | `str` | 0 | 0.00% |
| `transcript` | `str` | 86 | 3.37% |
| `duration` | `int64` | 0 | 0.00% |
| `published_date` | `str` | 0 | 0.00% |
| `views` | `int64` | 0 | 0.00% |
| `comments` | `int64` | 0 | 0.00% |
| `ratings` | `str` | 0 | 0.00% |
| `event` | `str` | 0 | 0.00% |
| `url` | `str` | 0 | 0.00% |
| `source` | `str` | 0 | 0.00% |

## 3. Sample Records (First 3 Rows)
| talk_id | title | speaker | duration (s) | published_date | views |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | Do schools kill creativity? | Ken Robinson | 1164 | 2006-06-27 | 47227110 |
| 2 | Averting the climate crisis | Al Gore | 977 | 2006-06-27 | 3200520 |
| 3 | Simplicity sells | David Pogue | 1286 | 2006-06-27 | 1636292 |

## 4. Known Limitations & Module 2 Hand-off Notes
1. **Missing Transcripts (`86` records / `3.37%`):** Primarily musical performances, dance showcases, or visual-only talks where spoken transcripts do not exist. Will be imputed using `title + description + tags` or filtered in Module 2.
2. **Missing `speaker_occupation` (`6` records):** Minor metadata gaps for ensemble/group speakers; will be filled with `'Unknown Occupation'` in Module 2.
3. **Aggregate Crowd Ratings vs. Individual User History:** The `ratings` column provides aggregate crowd rating counts (e.g., *Inspiring*, *Fascinating*, *Informative*) rather than individual user watch logs, due to TED user privacy.