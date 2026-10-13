"""
Unit and API integration tests for Part 1 starter code fixes.
"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError, PyMongoError, ServerSelectionTimeoutError
import pytest

from app.main import (
    app,
    init_indexes,
    IST,
    epoch_ms_to_datetime,
    datetime_to_epoch_ms,
    compute_attendance_date,
    compute_late_minutes,
    compute_work_hours,
    compute_overtime,
    compute_half_day,
    EmployeeIn,
    PunchInIn,
    format_employee,
    format_attendance_record,
)

test_client = TestClient(app)


class TestIndexes(unittest.TestCase):
    @patch("app.main.db")
    def test_init_indexes_creates_required_indexes(self, mock_db):
        init_indexes()
        # Verify employee collection indexes
        mock_db.employees.create_index.assert_any_call([("emp_code", 1)], unique=True)
        mock_db.employees.create_index.assert_any_call([("department", 1), ("emp_code", 1)])
        # Verify attendance_logs collection indexes
        mock_db.attendance_logs.create_index.assert_any_call([("emp_code", 1), ("date", 1)], unique=True)
        mock_db.attendance_logs.create_index.assert_any_call([("date", -1), ("emp_code", 1)])

    @patch("app.main.db")
    def test_init_indexes_does_not_swallow_errors(self, mock_db):
        mock_db.employees.create_index.side_effect = PyMongoError("Index creation failed")
        with self.assertRaises(PyMongoError):
            init_indexes()


class TestHelpers(unittest.TestCase):
    def test_compute_late_minutes_grace_period(self):
        # Shift 09:30: 09:40:00 gives 0; 09:40:01 gives 10; 10:15:59 gives 45 (R2)
        dt_094000 = datetime(2026, 7, 6, 9, 40, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_094000, "09:30"), 0)

        dt_094001 = datetime(2026, 7, 6, 9, 40, 1, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_094001, "09:30"), 10)

        dt_101559 = datetime(2026, 7, 6, 10, 15, 59, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_101559, "09:30"), 45)

        # Early punch-in (09:15) gives 0
        dt_091500 = datetime(2026, 7, 6, 9, 15, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(dt_091500, "09:30"), 0)

    def test_compute_late_minutes_utc_conversion(self):
        # 10:05 IST is 04:35 UTC (late by 35 min)
        dt_utc = datetime(2026, 7, 6, 4, 35, 0, tzinfo=timezone.utc)
        self.assertEqual(compute_late_minutes(dt_utc, "09:30"), 35)

    def test_compute_late_minutes_overnight(self):
        # Overnight shift 22:00 - 06:00, date 2026-07-06
        # Punch at 21:55 IST -> 0
        p1 = datetime(2026, 7, 6, 21, 55, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(p1, "22:00", "2026-07-06"), 0)

        # Punch at 22:15 IST -> 15 min late
        p2 = datetime(2026, 7, 6, 22, 15, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(p2, "22:00", "2026-07-06"), 15)

        # Punch next morning at 00:05 IST (belongs to 2026-07-06 shift) -> 125 min late
        p3 = datetime(2026, 7, 7, 0, 5, 0, tzinfo=IST)
        self.assertEqual(compute_late_minutes(p3, "22:00", "2026-07-06"), 125)

    def test_compute_work_hours_half_up_rounding(self):
        # R4: rounded to 2 decimals half-up (not banker's rounding)
        # 27747 seconds / 3600 = 7.7075 hours. Banker's rounding gives 7.70; Half-up gives 7.71
        t_in = datetime(2026, 7, 6, 9, 0, 0, tzinfo=IST)
        t_out = t_in + timedelta(seconds=27747)
        self.assertEqual(compute_work_hours(t_in, t_out), 7.71)

        # Sample 1: 32820 seconds -> 9.12
        t_out_s1 = t_in + timedelta(seconds=32820)
        self.assertEqual(compute_work_hours(t_in, t_out_s1), 9.12)

        # Sample 2: 32700 seconds -> 9.08
        t_out_s2 = t_in + timedelta(seconds=32700)
        self.assertEqual(compute_work_hours(t_in, t_out_s2), 9.08)

        # Sample 4: 12600 seconds -> 3.50
        t_out_s4 = t_in + timedelta(seconds=12600)
        self.assertEqual(compute_work_hours(t_in, t_out_s4), 3.5)

        # Sample 7: 31500 seconds -> 8.75
        t_out_s7 = t_in + timedelta(seconds=31500)
        self.assertEqual(compute_work_hours(t_in, t_out_s7), 8.75)

    def test_compute_overtime(self):
        # R3: at least 30 minutes; otherwise 0
        t_out_5min = datetime(2026, 7, 6, 18, 35, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(t_out_5min, "18:30", "2026-07-06"), 0)

        t_out_29min = datetime(2026, 7, 6, 18, 59, 59, tzinfo=IST)
        self.assertEqual(compute_overtime(t_out_29min, "18:30", "2026-07-06"), 0)

        t_out_30min = datetime(2026, 7, 6, 19, 0, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(t_out_30min, "18:30", "2026-07-06"), 30)

        t_out_40min = datetime(2026, 7, 6, 19, 10, 0, tzinfo=IST)
        self.assertEqual(compute_overtime(t_out_40min, "18:30", "2026-07-06"), 40)

        # Overnight shift 22:00 - 06:00
        # Out at 06:40 IST on next day -> 40 min overtime
        t_out_overnight = datetime(2026, 7, 7, 6, 40, 0, tzinfo=IST)
        self.assertEqual(
            compute_overtime(t_out_overnight, "06:00", "2026-07-06", shift_start="22:00"),
            40,
        )

    def test_compute_half_day(self):
        # R5: half_day is true when work_hours < 4.50
        self.assertTrue(compute_half_day(3.50))
        self.assertTrue(compute_half_day(4.49))
        self.assertFalse(compute_half_day(4.50))
        self.assertFalse(compute_half_day(9.12))
        self.assertFalse(compute_half_day(None))

    def test_compute_attendance_date(self):
        # Normal shift 09:30 - 18:30
        p_normal = datetime(2026, 7, 6, 9, 28, 0, tzinfo=IST)
        self.assertEqual(compute_attendance_date(p_normal, "09:30", "18:30"), "2026-07-06")

        # Overnight shift 22:00 - 06:00
        # Punch at 21:55 IST on 2026-07-06 -> belongs to 2026-07-06
        p_overnight_1 = datetime(2026, 7, 6, 21, 55, 0, tzinfo=IST)
        self.assertEqual(compute_attendance_date(p_overnight_1, "22:00", "06:00"), "2026-07-06")

        # Punch at 01:10 IST on 2026-07-07 -> earlier than shift_end 06:00 -> belongs to 2026-07-06
        p_overnight_2 = datetime(2026, 7, 7, 1, 10, 0, tzinfo=IST)
        self.assertEqual(compute_attendance_date(p_overnight_2, "22:00", "06:00"), "2026-07-06")

        # Punch at 06:00 IST on 2026-07-07 -> not earlier than shift_end -> belongs to 2026-07-07
        p_overnight_3 = datetime(2026, 7, 7, 6, 0, 0, tzinfo=IST)
        self.assertEqual(compute_attendance_date(p_overnight_3, "22:00", "06:00"), "2026-07-07")

    def test_epoch_conversion(self):
        # 1783310280900 ms should truncate to 1783310280 seconds (2026-07-06T03:58:00Z)
        dt = epoch_ms_to_datetime(1783310280900)
        self.assertEqual(dt, datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc))
        self.assertEqual(datetime_to_epoch_ms(dt), 1783310280000)


class TestValidation(unittest.TestCase):
    def test_employee_validation(self):
        # Valid employee with regular shift
        emp = EmployeeIn(
            emp_code="EMP0001",
            name="Asha Rao",
            email="asha@example.com",
            department="Engineering",
            shift_start="09:30",
            shift_end="18:30",
            joined_on="2026-01-05",
        )
        self.assertEqual(emp.emp_code, "EMP0001")

        # Valid employee with overnight shift (shift_end < shift_start)
        emp_overnight = EmployeeIn(
            emp_code="EMP0005",
            name="Karan Verma",
            email="karan@example.com",
            department="Operations",
            shift_start="22:00",
            shift_end="06:00",
            joined_on="2026-01-05",
        )
        self.assertEqual(emp_overnight.shift_start, "22:00")
        self.assertEqual(emp_overnight.shift_end, "06:00")

        # Invalid emp_code
        with self.assertRaises(Exception):
            EmployeeIn(
                emp_code="INVALID",
                name="Asha Rao",
                email="asha@example.com",
                department="Engineering",
                joined_on="2026-01-05",
            )

        # Same shift_start and shift_end (explicitly forbidden by openapi.yaml: shift_start must differ from shift_end)
        with self.assertRaises(Exception):
            EmployeeIn(
                emp_code="EMP0001",
                name="Asha Rao",
                email="asha@example.com",
                department="Engineering",
                shift_start="09:30",
                shift_end="09:30",
                joined_on="2026-01-05",
            )

        # Invalid email
        with self.assertRaises(Exception):
            EmployeeIn(
                emp_code="EMP0001",
                name="Asha Rao",
                email="not-an-email",
                department="Engineering",
                joined_on="2026-01-05",
            )

        # Invalid calendar date (e.g. Feb 30)
        with self.assertRaises(Exception):
            EmployeeIn(
                emp_code="EMP0001",
                name="Asha Rao",
                email="asha@example.com",
                department="Engineering",
                joined_on="2026-02-30",
            )

    def test_punch_in_validation(self):
        # Valid punch in
        p = PunchInIn(emp_code="EMP0001", status="PRESENT")
        self.assertEqual(p.status, "PRESENT")

        # Invalid status
        with self.assertRaises(Exception):
            PunchInIn(emp_code="EMP0001", status="ABSENT")

        # Invalid punched_at (seconds instead of ms)
        with self.assertRaises(Exception):
            PunchInIn(emp_code="EMP0001", punched_at=1783310280)

        # Invalid punched_at (float)
        with self.assertRaises(Exception):
            PunchInIn(emp_code="EMP0001", punched_at=1783310280000.5)


class TestEndpoints(unittest.TestCase):
    def test_health_endpoint(self):
        with patch("app.main.client") as mock_client:
            mock_client.admin.command.return_value = {"ok": 1}
            resp = test_client.get("/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), {"status": "ok"})

            mock_client.admin.command.side_effect = ServerSelectionTimeoutError("mongo down")
            resp = test_client.get("/health")
            self.assertEqual(resp.status_code, 503)

    def test_create_employee_success_and_conflict(self):
        payload = {
            "emp_code": "EMP0001",
            "name": "Asha Rao",
            "email": "asha@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-05",
        }
        with patch("app.main.db") as mock_db:
            mock_db.employees.insert_one.return_value = MagicMock(inserted_id="fake_id")
            resp = test_client.post("/employees", json=payload)
            self.assertEqual(resp.status_code, 201)
            data = resp.json()
            self.assertEqual(data["emp_code"], "EMP0001")
            self.assertNotIn("_id", data)
            self.assertNotIn("id", data)
            self.assertIsInstance(data["created_at"], int)

            # Duplicate emp_code handled atomically via DuplicateKeyError
            mock_db.employees.insert_one.side_effect = DuplicateKeyError("E11000 duplicate key")
            resp_conflict = test_client.post("/employees", json=payload)
            self.assertEqual(resp_conflict.status_code, 409)

    def test_list_employees_pagination_and_filter(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.count_documents.return_value = 1
            mock_cursor = MagicMock()
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.skip.return_value = mock_cursor
            mock_cursor.limit.return_value = [
                {
                    "emp_code": "EMP0001",
                    "name": "Asha Rao",
                    "email": "asha@example.com",
                    "department": "Engineering",
                    "shift_start": "09:30",
                    "shift_end": "18:30",
                    "joined_on": "2026-01-05",
                    "created_at": datetime(2026, 1, 5, 4, 30, tzinfo=timezone.utc),
                }
            ]
            mock_db.employees.find.return_value = mock_cursor

            resp = test_client.get("/employees?department=Engineering&page=1&page_size=20")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["total"], 1)
            self.assertEqual(data["page"], 1)
            self.assertEqual(data["page_size"], 20)
            self.assertEqual(len(data["items"]), 1)
            self.assertEqual(data["items"][0]["emp_code"], "EMP0001")
            self.assertIsInstance(data["items"][0]["created_at"], int)

            # Check skip calculation: page 1 must skip 0
            mock_cursor.skip.assert_called_with(0)
            mock_cursor.sort.assert_called_with("emp_code", 1)
            mock_db.employees.count_documents.assert_called_with({"department": "Engineering"})

    def test_list_employees_pagination_boundaries(self):
        # page < 1 -> 422
        resp_page_0 = test_client.get("/employees?page=0")
        self.assertEqual(resp_page_0.status_code, 422)

        # page_size < 1 -> 422
        resp_size_0 = test_client.get("/employees?page_size=0")
        self.assertEqual(resp_size_0.status_code, 422)

        # page_size > 100 -> 422
        resp_size_101 = test_client.get("/employees?page_size=101")
        self.assertEqual(resp_size_101.status_code, 422)

    def test_punch_in_endpoint(self):
        with patch("app.main.db") as mock_db:
            # 1. Nonexistent employee -> 404
            mock_db.employees.find_one.return_value = None
            resp = test_client.post("/attendance/punch-in", json={"emp_code": "EMP9999"})
            self.assertEqual(resp.status_code, 404)

            # 2. Existing employee -> 201
            mock_db.employees.find_one.return_value = {
                "emp_code": "EMP0001",
                "shift_start": "09:30",
                "shift_end": "18:30",
            }
            mock_db.attendance_logs.insert_one.return_value = MagicMock(inserted_id="fake_id")
            resp = test_client.post(
                "/attendance/punch-in",
                json={"emp_code": "EMP0001", "punched_at": 1783310280000, "status": "PRESENT"},
            )
            self.assertEqual(resp.status_code, 201)
            data = resp.json()
            self.assertEqual(data["emp_code"], "EMP0001")
            self.assertEqual(data["date"], "2026-07-06")
            self.assertEqual(data["status"], "PRESENT")
            self.assertEqual(data["punch_in"], 1783310280000)
            self.assertIsNone(data["punch_out"])
            self.assertIsNone(data["work_hours"])
            self.assertEqual(data["late_minutes"], 0)
            self.assertEqual(data["overtime_minutes"], 0)
            self.assertFalse(data["half_day"])
            self.assertEqual(data["history"], [])
            self.assertNotIn("_id", data)
            self.assertNotIn("id", data)

            # 3. Duplicate punch-in for same day -> 409
            mock_db.attendance_logs.insert_one.side_effect = DuplicateKeyError("duplicate key")
            resp_dup = test_client.post(
                "/attendance/punch-in",
                json={"emp_code": "EMP0001", "punched_at": 1783310280000},
            )
            self.assertEqual(resp_dup.status_code, 409)

    def test_list_attendance_endpoint(self):
        # Validation: date_from > date_to -> 422
        resp = test_client.get("/attendance?date_from=2026-07-10&date_to=2026-07-01")
        self.assertEqual(resp.status_code, 422)

        # Validation: invalid date format -> 422
        resp_invalid_date = test_client.get("/attendance?date_from=not-a-date")
        self.assertEqual(resp_invalid_date.status_code, 422)

        # Happy path
        with patch("app.main.db") as mock_db:
            mock_db.attendance_logs.count_documents.return_value = 1
            mock_cursor = MagicMock()
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.skip.return_value = mock_cursor
            mock_cursor.limit.return_value = [
                {
                    "emp_code": "EMP0001",
                    "date": "2026-07-06",
                    "status": "PRESENT",
                    "punch_in": datetime(2026, 7, 6, 3, 58, tzinfo=timezone.utc),
                    "punch_out": datetime(2026, 7, 6, 13, 5, tzinfo=timezone.utc),
                    "work_hours": 9.12,
                    "late_minutes": 0,
                    "overtime_minutes": 0,
                    "half_day": False,
                    "history": [],
                }
            ]
            mock_db.attendance_logs.find.return_value = mock_cursor

            resp = test_client.get("/attendance?emp_code=EMP0001&page=1&page_size=20")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["total"], 1)
            self.assertEqual(len(data["items"]), 1)
            rec = data["items"][0]
            self.assertEqual(rec["punch_in"], 1783310280000)
            self.assertEqual(rec["punch_out"], 1783343100000)
            self.assertNotIn("_id", rec)
            self.assertNotIn("id", rec)

            # Verify database sorting on (date: -1, emp_code: 1) and skip
            mock_cursor.sort.assert_called_with([("date", -1), ("emp_code", 1)])
            mock_cursor.skip.assert_called_with(0)
