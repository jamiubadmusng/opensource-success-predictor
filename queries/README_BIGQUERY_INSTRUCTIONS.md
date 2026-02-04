# BigQuery Instructions for GH Archive Data Extraction

## Overview

This folder contains SQL queries to extract data from the GitHub Archive (GH Archive) public dataset on Google BigQuery. The queries are designed to build a dataset for predicting open-source project success.

## Query Execution Order

For the complete dataset, you only need to run **Query 5** (`05_combined_dataset.sql`). The other queries are provided for reference and can be run individually if you want to explore specific metrics.

| Query | Purpose | Estimated Data Scanned |
|-------|---------|----------------------|
| 01 | Extract new repositories | ~50-100 GB |
| 02 | Early activity metrics | ~100-150 GB |
| 03 | Outcome metrics | ~150-200 GB |
| 04 | Cumulative stars | ~100 GB |
| **05** | **Combined dataset (USE THIS)** | **~200-300 GB** |

## Step-by-Step Instructions

### Step 1: Open BigQuery Console

1. Go to [BigQuery Console](https://console.cloud.google.com/bigquery)
2. Make sure your project is selected in the top dropdown

### Step 2: Run the Combined Query

1. Click the **"+"** button to create a new query tab
2. Copy the entire contents of `05_combined_dataset.sql`
3. Paste it into the query editor
4. Click **"Run"**
5. Wait for the query to complete (may take 2-5 minutes)

### Step 3: Export Results to CSV

1. Once the query completes, you will see the results in the bottom panel
2. Click **"Save Results"** button above the results
3. Select **"CSV (local file)"** to download directly, OR
4. Select **"CSV (Google Drive)"** for larger files
5. Name the file `github_projects_dataset.csv`

### Step 4: Move the CSV to Your Project

1. Move the downloaded CSV file to:
   ```
   c:\Users\muham\portfolio\tech\data\raw\github_projects_dataset.csv
   ```

## Cost Estimation

- Free tier: 1 TB of query processing per month
- Free trial: $300 credit for 90 days
- Query 5 estimates: ~200-300 GB per run
- **You can run the main query 3-4 times within the free monthly limit**

## Expected Output

The combined query will return approximately 50,000-200,000 rows with the following columns:

### Repository Information
- `repo_id`, `repo_name`, `primary_language`, `description`
- `repo_created_at`, `creator_id`, `creator_login`

### Early Activity Features (First 30 Days)
- `push_events_30d`, `total_commits_30d`
- `issues_opened_30d`, `issues_closed_30d`, `issue_comments_30d`
- `prs_opened_30d`, `prs_closed_30d`, `pr_reviews_30d`
- `stars_30d`, `forks_30d`
- `unique_contributors_30d`, `unique_committers_30d`
- `events_week_1` through `events_week_4`

### Outcome Metrics (6 Months)
- `total_stars_6m`, `total_forks_6m`
- `total_contributors_6m`, `total_committers_6m`
- `pushes_months_4_to_6`, `issues_months_4_to_6`, `prs_months_4_to_6`
- `total_releases_6m`, `total_events_6m`

### Target Variables
- `is_starred_project` (>=100 stars)
- `has_community` (>=5 contributors)
- `is_actively_maintained` (commits in months 4-6)
- `is_successful_project` (combined success metric)

## Troubleshooting

### Query takes too long
- BigQuery queries on large datasets can take 2-10 minutes
- If it exceeds 10 minutes, try running during off-peak hours

### "Quota exceeded" error
- You have exceeded the free tier limit
- Wait until the next month or use your $300 credit

### Results too large to download
- Export to Google Drive instead of local download
- Or add `LIMIT 100000` at the end of the query

## Next Steps

After downloading the CSV:
1. Place it in `tech/data/raw/`
2. Return to the conversation to continue with the Jupyter Notebook analysis
