# Employee Attendance & Analytics API

A FastAPI and MongoDB backend service for managing employee attendance, punch-ins/outs, manual regularizations, and departmental analytics.

## Setup and Running

1. **Virtual environment and dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment configuration:**
   Set `MONGO_URI` and `MONGO_DB` environment variables (defaults to `mongodb://localhost:27017` and `attendance_db`).

3. **Database seeding (optional sample dataset):**
   ```bash
   python sample_seed.py
   ```

4. **Start the API:**
   ```bash
   uvicorn app.main:app --port 8000
   ```

5. **Run test suite:**
   ```bash
   pytest tests -v
   ```

## Architecture and Highlights

- **FastAPI + MongoDB (PyMongo):** All endpoints and logic are self-contained in `app/main.py`.
- **Database-enforced Concurrency:** Unique indexes on `employees.emp_code` and `attendance_logs.(emp_code, date)` guarantee race safety during concurrent punch-ins and creations.
- **Optimistic Concurrency Control:** Manual regularizations use atomic size checks on `history` to avoid lost audit trail updates under concurrent modifications.
- **Analytics Aggregations:** Complex reporting endpoints (`monthly`, `summary`, `leaderboard`, `trend`) are executed entirely within MongoDB aggregation pipelines using `$lookup`, `$densify`, and `$setWindowFields`.
- **Index-backed Queries:** Indexes are initialized idempotently on startup, supporting IXSCAN query plans across listing and analytics queries.
