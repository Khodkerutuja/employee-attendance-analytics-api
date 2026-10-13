# DECISIONS.md

1. **Indexes.**
   - `employees`: unique `{"emp_code": 1}` guarantees employee code uniqueness; `{"department": 1, "emp_code": 1}` optimizes filtered employee listing; `{"department": 1, "joined_on": 1}` and `{"joined_on": 1}` support department headcount aggregations.
   - `attendance_logs`: unique `{"emp_code": 1, "date": 1}` enforces one daily record and race-safe punch-ins; `{"date": -1, "emp_code": 1}` serves attendance listing with sort; `{"emp_code": 1, "punch_in": -1}` serves punch-out lookups; `{"date": 1, "late_minutes": 1}` supports leaderboard matching.
   - *Rejected:* Standalone index on `status`, rejected because of low cardinality (only five status values).

2. **Punch-in race.**
   Both requests check employee existence and attempt an atomic `insert_one` against the unique index on `(emp_code, date)`. MongoDB commits the first insert (201). The second concurrent insert immediately raises a `DuplicateKeyError`, which our handler intercepts and returns as HTTP 409 Conflict.

3. **Ties.**
   Using `$setWindowFields` with `$rank`, tied employees share the same rank and subsequent ranks are skipped (e.g., 1, 2, 2, 4). We filter `rank <= limit` post-ranking, so all employees tied at the cutoff are included, ordered by `emp_code` ascending for determinism.

4. **Headcount.**
   The aggregation pipeline starts on `employees` (filtering `joined_on <= month_end`), grouping to count headcount before performing a `$lookup` into `attendance_logs`. Employees with zero attendance logs are thus preserved and counted.

5. **One thing to change for 100x data.**
   Shard `attendance_logs` by `emp_code` to horizontally distribute write load, and maintain incremental daily summary rollups to eliminate heavy month-wide scans during analytics queries.
