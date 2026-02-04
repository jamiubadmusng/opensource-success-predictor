-- =============================================================================
-- Query 2: Early Activity Metrics (First 30 Days)
-- Purpose: Capture activity signals in the first month after repository creation
-- Estimated data scanned: ~100-150 GB
-- =============================================================================

-- This query calculates key early-stage metrics for repositories created in Q1 2024
-- These metrics serve as predictive features for project success

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

early_events AS (
    -- Get all events for these repos in their first 30 days
    SELECT
        e.repo.id AS repo_id,
        e.repo.name AS repo_name,
        e.type AS event_type,
        e.actor.id AS actor_id,
        e.actor.login AS actor_login,
        e.created_at AS event_time,
        e.payload,
        nr.repo_created_at
    FROM
        `githubarchive.day.2024*` e
    INNER JOIN
        new_repos nr ON e.repo.id = nr.repo_id
    WHERE
        e._TABLE_SUFFIX BETWEEN '0101' AND '0430'
        AND TIMESTAMP_DIFF(e.created_at, nr.repo_created_at, DAY) BETWEEN 0 AND 30
)

SELECT
    repo_id,
    repo_name,
    repo_created_at,
    
    -- Commit activity
    COUNTIF(event_type = 'PushEvent') AS push_events_30d,
    SUM(CASE 
        WHEN event_type = 'PushEvent' 
        THEN CAST(JSON_EXTRACT_SCALAR(payload, '$.size') AS INT64) 
        ELSE 0 
    END) AS total_commits_30d,
    
    -- Issue activity
    COUNTIF(event_type = 'IssuesEvent') AS issue_events_30d,
    COUNTIF(event_type = 'IssuesEvent' 
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened') AS issues_opened_30d,
    COUNTIF(event_type = 'IssuesEvent' 
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed') AS issues_closed_30d,
    COUNTIF(event_type = 'IssueCommentEvent') AS issue_comments_30d,
    
    -- Pull request activity
    COUNTIF(event_type = 'PullRequestEvent') AS pr_events_30d,
    COUNTIF(event_type = 'PullRequestEvent' 
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened') AS prs_opened_30d,
    COUNTIF(event_type = 'PullRequestEvent' 
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed') AS prs_closed_30d,
    COUNTIF(event_type = 'PullRequestReviewEvent') AS pr_reviews_30d,
    COUNTIF(event_type = 'PullRequestReviewCommentEvent') AS pr_review_comments_30d,
    
    -- Community engagement
    COUNTIF(event_type = 'WatchEvent') AS stars_30d,
    COUNTIF(event_type = 'ForkEvent') AS forks_30d,
    
    -- Contributor metrics
    COUNT(DISTINCT actor_id) AS unique_contributors_30d,
    COUNT(DISTINCT CASE 
        WHEN event_type = 'PushEvent' THEN actor_id 
    END) AS unique_committers_30d,
    COUNT(DISTINCT CASE 
        WHEN event_type IN ('IssuesEvent', 'IssueCommentEvent') THEN actor_id 
    END) AS unique_issue_participants_30d,
    
    -- Activity distribution
    COUNT(*) AS total_events_30d,
    COUNTIF(TIMESTAMP_DIFF(event_time, repo_created_at, DAY) <= 7) AS events_first_week,
    COUNTIF(TIMESTAMP_DIFF(event_time, repo_created_at, DAY) BETWEEN 8 AND 14) AS events_second_week,
    COUNTIF(TIMESTAMP_DIFF(event_time, repo_created_at, DAY) BETWEEN 15 AND 21) AS events_third_week,
    COUNTIF(TIMESTAMP_DIFF(event_time, repo_created_at, DAY) BETWEEN 22 AND 30) AS events_fourth_week

FROM
    early_events
GROUP BY
    repo_id, repo_name, repo_created_at
HAVING
    total_events_30d >= 5  -- Filter out completely inactive repos
