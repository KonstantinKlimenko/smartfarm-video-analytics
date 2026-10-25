CREATE TABLE IF NOT EXISTS crossing_events (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_name TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    time_seconds NUMERIC(10, 3) NOT NULL,
    direction TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);