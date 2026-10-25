CREATE TABLE IF NOT EXISTS analysis_runs (
    id UUID PRIMARY KEY,
    source_name TEXT NOT NULL,
    weights TEXT NOT NULL CHECK (weights IN ('base', 'v2', 'v3')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    output_video TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

ALTER TABLE crossing_events
ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES analysis_runs(id);