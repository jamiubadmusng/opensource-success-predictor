-- =============================================================================
-- Query 5: Combined Dataset for Machine Learning
-- Purpose: Single query to extract all features and target variables
-- THIS IS THE MAIN QUERY TO RUN - Combines all metrics into one dataset
-- Estimated data scanned: ~200-300 GB (within free tier $300 credit)
-- =============================================================================

WITH new_repos AS (
    -- Step 1: Identify non-fork repositories created in Q1 2024
    SELECT
        repo.id AS repo_id,
        repo.name AS repo_name,
        JSON_EXTRACT_SCALAR(payload, '$.repository.language') AS primary_language,
        JSON_EXTRACT_SCALAR(payload, '$.repository.description') AS description,
        MIN(created_at) AS repo_created_at,
        actor.id AS creator_id,
        actor.login AS creator_login
    FROM
        `githubarchive.day.2024*`
    WHERE
        _TABLE_SUFFIX BETWEEN '0101' AND '0331'
        AND type = 'CreateEvent'
        AND JSON_EXTRACT_SCALAR(payload, '$.ref_type') = 'repository'
    GROUP BY
        repo.id, repo.name, primary_language, description, actor.id, actor.login
),

all_events AS (
    -- Step 2: Get all events for these repos over 6 months
    SELECT
        e.repo.id AS repo_id,
        e.type AS event_type,
        e.actor.id AS actor_id,
        e.created_at AS event_time,
        e.payload,
        nr.repo_created_at,
        TIMESTAMP_DIFF(e.created_at, nr.repo_created_at, DAY) AS days_since_creation
    FROM
        `githubarchive.day.2024*` e
    INNER JOIN
        new_repos nr ON e.repo.id = nr.repo_id
    WHERE
        e._TABLE_SUFFIX BETWEEN '0101' AND '0930'
        AND TIMESTAMP_DIFF(e.created_at, nr.repo_created_at, DAY) BETWEEN 0 AND 180
),

early_metrics AS (
    -- Step 3: Calculate first 30 days metrics (FEATURES)
    SELECT
        repo_id,
        
        -- Commit activity (first 30 days)
        COUNTIF(event_type = 'PushEvent' AND days_since_creation <= 30) AS push_events_30d,
        SUM(CASE 
            WHEN event_type = 'PushEvent' AND days_since_creation <= 30
            THEN COALESCE(CAST(JSON_EXTRACT_SCALAR(payload, '$.size') AS INT64), 0)
            ELSE 0 
        END) AS total_commits_30d,
        
        -- Issue activity (first 30 days)
        COUNTIF(event_type = 'IssuesEvent' AND days_since_creation <= 30) AS issue_events_30d,
        COUNTIF(event_type = 'IssuesEvent' 
            AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
            AND days_since_creation <= 30) AS issues_opened_30d,
        COUNTIF(event_type = 'IssuesEvent' 
            AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
            AND days_since_creation <= 30) AS issues_closed_30d,
        COUNTIF(event_type = 'IssueCommentEvent' AND days_since_creation <= 30) AS issue_comments_30d,
        
        -- Pull request activity (first 30 days)
        COUNTIF(event_type = 'PullRequestEvent' AND days_since_creation <= 30) AS pr_events_30d,
        COUNTIF(event_type = 'PullRequestEvent' 
            AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'opened'
            AND days_since_creation <= 30) AS prs_opened_30d,
        COUNTIF(event_type = 'PullRequestEvent' 
            AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
            AND days_since_creation <= 30) AS prs_closed_30d,
        COUNTIF(event_type = 'PullRequestReviewEvent' AND days_since_creation <= 30) AS pr_reviews_30d,
        
        -- Community engagement (first 30 days)
        COUNTIF(event_type = 'WatchEvent' AND days_since_creation <= 30) AS stars_30d,
        COUNTIF(event_type = 'ForkEvent' AND days_since_creation <= 30) AS forks_30d,
        
        -- Contributor metrics (first 30 days)
        COUNT(DISTINCT CASE WHEN days_since_creation <= 30 THEN actor_id END) AS unique_contributors_30d,
        COUNT(DISTINCT CASE 
            WHEN event_type = 'PushEvent' AND days_since_creation <= 30 THEN actor_id 
        END) AS unique_committers_30d,
        
        -- Weekly activity distribution (first 30 days)
        COUNTIF(days_since_creation <= 7) AS events_week_1,
        COUNTIF(days_since_creation BETWEEN 8 AND 14) AS events_week_2,
        COUNTIF(days_since_creation BETWEEN 15 AND 21) AS events_week_3,
        COUNTIF(days_since_creation BETWEEN 22 AND 30) AS events_week_4,
        
        -- Total early activity
        COUNTIF(days_since_creation <= 30) AS total_events_30d
        
    FROM all_events
    GROUP BY repo_id
),

outcome_metrics AS (
    -- Step 4: Calculate 6-month outcomes (TARGET VARIABLES)
    SELECT
        repo_id,
        
        -- Cumulative metrics at 6 months
        COUNTIF(event_type = 'WatchEvent') AS total_stars_6m,
        COUNTIF(event_type = 'ForkEvent') AS total_forks_6m,
        COUNT(DISTINCT actor_id) AS total_contributors_6m,
        COUNT(DISTINCT CASE WHEN event_type = 'PushEvent' THEN actor_id END) AS total_committers_6m,
        
        -- Sustained activity (months 4-6)
        COUNTIF(event_type = 'PushEvent' AND days_since_creation > 90) AS pushes_months_4_to_6,
        COUNTIF(event_type = 'IssuesEvent' AND days_since_creation > 90) AS issues_months_4_to_6,
        COUNTIF(event_type = 'PullRequestEvent' AND days_since_creation > 90) AS prs_months_4_to_6,
        COUNT(DISTINCT CASE WHEN days_since_creation > 90 THEN actor_id END) AS contributors_months_4_to_6,
        
        -- Release maturity
        COUNTIF(event_type = 'ReleaseEvent') AS total_releases_6m,
        
        -- Total activity
        COUNT(*) AS total_events_6m
        
    FROM all_events
    GROUP BY repo_id
)

-- Final combined dataset
SELECT
    nr.repo_id,
    nr.repo_name,
    nr.primary_language,
    nr.description,
    nr.repo_created_at,
    nr.creator_id,
    nr.creator_login,
    
    -- Early activity features (first 30 days)
    COALESCE(em.push_events_30d, 0) AS push_events_30d,
    COALESCE(em.total_commits_30d, 0) AS total_commits_30d,
    COALESCE(em.issue_events_30d, 0) AS issue_events_30d,
    COALESCE(em.issues_opened_30d, 0) AS issues_opened_30d,
    COALESCE(em.issues_closed_30d, 0) AS issues_closed_30d,
    COALESCE(em.issue_comments_30d, 0) AS issue_comments_30d,
    COALESCE(em.pr_events_30d, 0) AS pr_events_30d,
    COALESCE(em.prs_opened_30d, 0) AS prs_opened_30d,
    COALESCE(em.prs_closed_30d, 0) AS prs_closed_30d,
    COALESCE(em.pr_reviews_30d, 0) AS pr_reviews_30d,
    COALESCE(em.stars_30d, 0) AS stars_30d,
    COALESCE(em.forks_30d, 0) AS forks_30d,
    COALESCE(em.unique_contributors_30d, 0) AS unique_contributors_30d,
    COALESCE(em.unique_committers_30d, 0) AS unique_committers_30d,
    COALESCE(em.events_week_1, 0) AS events_week_1,
    COALESCE(em.events_week_2, 0) AS events_week_2,
    COALESCE(em.events_week_3, 0) AS events_week_3,
    COALESCE(em.events_week_4, 0) AS events_week_4,
    COALESCE(em.total_events_30d, 0) AS total_events_30d,
    
    -- Outcome metrics (6 months)
    COALESCE(om.total_stars_6m, 0) AS total_stars_6m,
    COALESCE(om.total_forks_6m, 0) AS total_forks_6m,
    COALESCE(om.total_contributors_6m, 0) AS total_contributors_6m,
    COALESCE(om.total_committers_6m, 0) AS total_committers_6m,
    COALESCE(om.pushes_months_4_to_6, 0) AS pushes_months_4_to_6,
    COALESCE(om.issues_months_4_to_6, 0) AS issues_months_4_to_6,
    COALESCE(om.prs_months_4_to_6, 0) AS prs_months_4_to_6,
    COALESCE(om.contributors_months_4_to_6, 0) AS contributors_months_4_to_6,
    COALESCE(om.total_releases_6m, 0) AS total_releases_6m,
    COALESCE(om.total_events_6m, 0) AS total_events_6m,
    
    -- Derived success indicators (for target variable creation)
    CASE 
        WHEN COALESCE(om.total_stars_6m, 0) >= 100 THEN 1 
        ELSE 0 
    END AS is_starred_project,
    CASE 
        WHEN COALESCE(om.total_contributors_6m, 0) >= 5 THEN 1 
        ELSE 0 
    END AS has_community,
    CASE 
        WHEN COALESCE(om.pushes_months_4_to_6, 0) > 0 THEN 1 
        ELSE 0 
    END AS is_actively_maintained,
    -- Combined success metric: starred OR has community AND still maintained
    CASE 
        WHEN COALESCE(om.total_stars_6m, 0) >= 50 
            OR (COALESCE(om.total_contributors_6m, 0) >= 3 
                AND COALESCE(om.pushes_months_4_to_6, 0) > 0)
        THEN 1 
        ELSE 0 
    END AS is_successful_project

FROM new_repos nr
LEFT JOIN early_metrics em ON nr.repo_id = em.repo_id
LEFT JOIN outcome_metrics om ON nr.repo_id = om.repo_id
WHERE
    -- Filter for repos with at least some early activity
    COALESCE(em.total_events_30d, 0) >= 3
ORDER BY
    om.total_stars_6m DESC NULLS LAST
