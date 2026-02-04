-- =============================================================================
-- Query 4: Cumulative Star Count Over Time
-- Purpose: Track total stars accumulated by 6 months for success classification
-- Estimated data scanned: ~100 GB
-- =============================================================================

WITH new_repos AS (
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
)

SELECT
    nr.repo_id,
    nr.repo_name,
    nr.repo_created_at,
    COUNTIF(e.type = 'WatchEvent') AS total_stars_6_months,
    COUNTIF(e.type = 'ForkEvent') AS total_forks_6_months,
    COUNT(DISTINCT CASE WHEN e.type = 'PushEvent' THEN e.actor.id END) AS total_unique_committers_6_months,
    COUNT(DISTINCT e.actor.id) AS total_unique_contributors_6_months
FROM
    new_repos nr
LEFT JOIN
    `githubarchive.day.2024*` e 
    ON nr.repo_id = e.repo.id
    AND e._TABLE_SUFFIX BETWEEN '0101' AND '0930'
    AND TIMESTAMP_DIFF(e.created_at, nr.repo_created_at, DAY) BETWEEN 0 AND 180
GROUP BY
    nr.repo_id, nr.repo_name, nr.repo_created_at
