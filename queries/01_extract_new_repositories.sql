-- =============================================================================
-- Query 1: Extract New Repositories Created in 2024
-- Purpose: Identify repositories created in early 2024 to track their growth
-- Estimated data scanned: ~50-100 GB (within free tier)
-- =============================================================================

-- This query extracts repositories created in Q1 2024 (Jan-Mar)
-- We use this window so we have sufficient time to observe their growth trajectory

SELECT
    repo.id AS repo_id,
    repo.name AS repo_name,
    JSON_EXTRACT_SCALAR(payload, '$.repository.language') AS primary_language,
    JSON_EXTRACT_SCALAR(payload, '$.repository.description') AS description,
    JSON_EXTRACT_SCALAR(payload, '$.repository.license.key') AS license,
    CAST(JSON_EXTRACT_SCALAR(payload, '$.repository.fork') AS BOOL) AS is_fork,
    created_at AS repo_created_at,
    actor.id AS creator_id,
    actor.login AS creator_login
FROM
    `githubarchive.day.2024*`
WHERE
    _TABLE_SUFFIX BETWEEN '0101' AND '0331'
    AND type = 'CreateEvent'
    AND JSON_EXTRACT_SCALAR(payload, '$.ref_type') = 'repository'
    -- Exclude forks to focus on original projects
    AND COALESCE(CAST(JSON_EXTRACT_SCALAR(payload, '$.repository.fork') AS BOOL), FALSE) = FALSE
GROUP BY
    repo.id, repo.name, primary_language, description, license, is_fork, created_at, actor.id, actor.login
