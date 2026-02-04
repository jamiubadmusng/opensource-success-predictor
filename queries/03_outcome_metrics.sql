-- =============================================================================
-- Query 3: Outcome Metrics (6 Months Later)
-- Purpose: Measure project success indicators 6 months after creation
-- These metrics define our target variable for prediction
-- Estimated data scanned: ~150-200 GB
-- =============================================================================

WITH new_repos AS (
    -- Get repositories created in Q1 2024
    SELECT DISTINCT
        repo.id AS repo_id,
        repo.name AS repo_name,
        MIN(created_at) AS repo_created_at
    FROM
        `githubarchive.day.2024*`
    WHERE
        _TABLE_SUFFIX BETWEEN '0101' AND '0331'
        AND type = 'CreateEvent'
        AND JSON_EXTRACT_SCALAR(payload, '$.ref_type') = 'repository'
    GROUP BY
        repo.id, repo.name
),

six_month_events AS (
    -- Get events from months 4-6 (days 90-180) to measure sustained activity
    SELECT
        e.repo.id AS repo_id,
        e.repo.name AS repo_name,
        e.type AS event_type,
        e.actor.id AS actor_id,
        e.created_at AS event_time,
        e.payload,
        nr.repo_created_at
    FROM
        `githubarchive.day.2024*` e
    INNER JOIN
        new_repos nr ON e.repo.id = nr.repo_id
    WHERE
        e._TABLE_SUFFIX BETWEEN '0401' AND '0930'
        AND TIMESTAMP_DIFF(e.created_at, nr.repo_created_at, DAY) BETWEEN 90 AND 180
)

SELECT
    repo_id,
    repo_name,
    repo_created_at,
    
    -- Cumulative stars and forks at 6 months
    COUNTIF(event_type = 'WatchEvent') AS stars_months_4_to_6,
    COUNTIF(event_type = 'ForkEvent') AS forks_months_4_to_6,
    
    -- Sustained development activity
    COUNTIF(event_type = 'PushEvent') AS push_events_months_4_to_6,
    SUM(CASE 
        WHEN event_type = 'PushEvent' 
        THEN CAST(JSON_EXTRACT_SCALAR(payload, '$.size') AS INT64) 
        ELSE 0 
    END) AS commits_months_4_to_6,
    
    -- Community health indicators
    COUNTIF(event_type = 'IssuesEvent') AS issue_events_months_4_to_6,
    COUNTIF(event_type = 'PullRequestEvent') AS pr_events_months_4_to_6,
    COUNTIF(event_type = 'PullRequestEvent' 
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened') AS external_prs_months_4_to_6,
    
    -- Contributor growth
    COUNT(DISTINCT actor_id) AS unique_contributors_months_4_to_6,
    COUNT(DISTINCT CASE 
        WHEN event_type = 'PushEvent' THEN actor_id 
    END) AS unique_committers_months_4_to_6,
    
    -- Total activity for success classification
    COUNT(*) AS total_events_months_4_to_6,
    
    -- Release activity (indicates maturity)
    COUNTIF(event_type = 'ReleaseEvent') AS releases_months_4_to_6

FROM
    six_month_events
GROUP BY
    repo_id, repo_name, repo_created_at
