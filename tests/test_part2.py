"""
Unit and integration tests for Part 2 endpoints.
"""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError, PyMongoError
import pytest

from app.main import (
    app,
    IST,
    count_working_days,
    round_2dp,
    round_4dp,
)

test_client = TestClient(app)


class TestPart2Helpers(unittest.TestCase):
    def test_count_working_days(self):
        # 2026-07 has 31 days (July 1 is Wed, July 31 is Fri). Total weekdays = 23.
        # Employee joined on 2026-07-06 (Monday). Weekdays from July 6 to July 31 = 20.
        days = count_working_days("2026-07", "2026-07-06")
        self.assertEqual(days, 20)

        # Mid-month joiner on Friday 2026-07-31 -> 1 working day
        days_last = count_working_days("2026-07", "2026-07-31")
        self.assertEqual(days_last, 1)

        # Joined after month end -> 0 working days
        days_after = count_working_days("2026-07", "2026-08-01")
        self.assertEqual(days_after, 0)

    def test_rounding_helpers(self):
        # 2 decimals half-up
        self.assertEqual(round_2dp(7.705), 7.71)
        self.assertEqual(round_2dp(7.704), 7.70)
        self.assertEqual(round_2dp(None), None)

        # 4 decimals half-up
        self.assertEqual(round_4dp(0.85715), 0.8572)
        self.assertEqual(round_4dp(0.85714), 0.8571)
        self.assertEqual(round_4dp(None), None)


class TestPunchOutEndpoint(unittest.TestCase):
    def test_punch_out_unknown_employee(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = None
            resp = test_client.post("/attendance/punch-out", json={"emp_code": "EMP9999"})
            self.assertEqual(resp.status_code, 404)

    def test_punch_out_no_punch_in(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = {"emp_code": "EMP0001"}
            mock_db.attendance_logs.find_one.return_value = None
            resp = test_client.post("/attendance/punch-out", json={"emp_code": "EMP0001"})
            self.assertEqual(resp.status_code, 404)

    def test_punch_out_already_punched_out(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = {"emp_code": "EMP0001"}
            mock_db.attendance_logs.find_one.return_value = {
                "_id": "fake_id",
                "emp_code": "EMP0001",
                "punch_in": datetime(2026, 7, 6, 4, 0, tzinfo=timezone.utc),
                "punch_out": datetime(2026, 7, 6, 13, 0, tzinfo=timezone.utc),
            }
            resp = test_client.post(
                "/attendance/punch-out",
                json={"emp_code": "EMP0001", "punched_at": 1783350000000},
            )
            self.assertEqual(resp.status_code, 409)

    def test_punch_out_before_punch_in(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = {"emp_code": "EMP0001"}
            mock_db.attendance_logs.find_one.return_value = {
                "_id": "fake_id",
                "emp_code": "EMP0001",
                "punch_in": datetime(2026, 7, 6, 10, 0, tzinfo=timezone.utc),
                "punch_out": None,
            }
            # punched_at earlier than punch_in
            resp = test_client.post(
                "/attendance/punch-out",
                json={"emp_code": "EMP0001", "punched_at": int(datetime(2026, 7, 6, 9, 0, tzinfo=timezone.utc).timestamp() * 1000)},
            )
            self.assertEqual(resp.status_code, 422)

    def test_punch_out_success_and_concurrency(self):
        with patch("app.main.db") as mock_db:
            p_in = datetime(2026, 7, 6, 3, 58, tzinfo=timezone.utc)
            mock_db.employees.find_one.return_value = {
                "emp_code": "EMP0001",
                "shift_start": "09:30",
                "shift_end": "18:30",
            }
            mock_db.attendance_logs.find_one.return_value = {
                "_id": "log_id",
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": p_in,
                "punch_out": None,
                "late_minutes": 0,
            }
            p_out = datetime(2026, 7, 6, 13, 5, tzinfo=timezone.utc)
            mock_db.attendance_logs.find_one_and_update.return_value = {
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": p_in,
                "punch_out": p_out,
                "work_hours": 9.12,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            }
            resp = test_client.post(
                "/attendance/punch-out",
                json={"emp_code": "EMP0001", "punched_at": int(p_out.timestamp() * 1000)},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["work_hours"], 9.12)
            self.assertEqual(data["overtime_minutes"], 0)
            self.assertFalse(data["half_day"])
            self.assertEqual(data["history"], [])

            # Concurrency: second simultaneous request gets None from find_one_and_update -> 409
            mock_db.attendance_logs.find_one_and_update.return_value = None
            resp_conflict = test_client.post(
                "/attendance/punch-out",
                json={"emp_code": "EMP0001", "punched_at": int(p_out.timestamp() * 1000)},
            )
            self.assertEqual(resp_conflict.status_code, 409)


class TestRegularizeEndpoint(unittest.TestCase):
    def test_regularize_unknown_employee(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = None
            resp = test_client.patch(
                "/attendance/EMP9999/2026-07-06",
                json={"reason": "biometric glitch", "regularized_by": "hr.admin"},
            )
            self.assertEqual(resp.status_code, 404)

    def test_regularize_no_record(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = {"emp_code": "EMP0001"}
            mock_db.attendance_logs.find_one.return_value = None
            resp = test_client.patch(
                "/attendance/EMP0001/2026-07-06",
                json={"reason": "biometric glitch", "regularized_by": "hr.admin"},
            )
            self.assertEqual(resp.status_code, 404)

    def test_regularize_absent_with_punches_422(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = {"emp_code": "EMP0001"}
            mock_db.attendance_logs.find_one.return_value = {
                "_id": "id1",
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 6, 4, 0, tzinfo=timezone.utc),
                "punch_out": None,
            }
            resp = test_client.patch(
                "/attendance/EMP0001/2026-07-06",
                json={
                    "status": "ABSENT",
                    "punch_in": 1783310280000,
                    "reason": "marked absent",
                    "regularized_by": "hr.admin",
                },
            )
            self.assertEqual(resp.status_code, 422)

    def test_regularize_changes_nothing_422(self):
        with patch("app.main.db") as mock_db:
            p_in = datetime(2026, 7, 6, 4, 0, tzinfo=timezone.utc)
            mock_db.employees.find_one.return_value = {
                "emp_code": "EMP0001",
                "shift_start": "09:30",
                "shift_end": "18:30",
            }
            mock_db.attendance_logs.find_one.return_value = {
                "_id": "id1",
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": p_in,
                "punch_out": None,
                "work_hours": None,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            }
            # Sending same punch_in and status
            resp = test_client.patch(
                "/attendance/EMP0001/2026-07-06",
                json={
                    "status": "PRESENT",
                    "punch_in": int(p_in.timestamp() * 1000),
                    "reason": "biometric check",
                    "regularized_by": "hr.admin",
                },
            )
            self.assertEqual(resp.status_code, 422)

    def test_regularize_success_and_history(self):
        with patch("app.main.db") as mock_db:
            old_in = datetime(2026, 7, 7, 4, 35, tzinfo=timezone.utc)
            new_in = datetime(2026, 7, 7, 3, 58, tzinfo=timezone.utc)
            p_out = datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
            mock_db.employees.find_one.return_value = {
                "emp_code": "EMP0001",
                "shift_start": "09:30",
                "shift_end": "18:30",
            }
            mock_db.attendance_logs.find_one.return_value = {
                "_id": "id1",
                "emp_code": "EMP0001",
                "date": "2026-07-07",
                "status": "PRESENT",
                "punch_in": old_in,
                "punch_out": p_out,
                "work_hours": 8.42,
                "late_minutes": 35,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            }
            mock_db.attendance_logs.find_one_and_update.return_value = {
                "emp_code": "EMP0001",
                "date": "2026-07-07",
                "status": "PRESENT",
                "punch_in": new_in,
                "punch_out": p_out,
                "work_hours": 9.03,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [
                    {
                        "at": datetime(2026, 7, 8, 5, 0, tzinfo=timezone.utc),
                        "by": "hr.admin",
                        "reason": "biometric glitch",
                        "changes": {
                            "punch_in": {"from": old_in, "to": new_in},
                            "late_minutes": {"from": 35, "to": 0},
                            "work_hours": {"from": 8.42, "to": 9.03},
                        },
                    }
                ],
            }
            resp = test_client.patch(
                "/attendance/EMP0001/2026-07-07",
                json={
                    "punch_in": int(new_in.timestamp() * 1000),
                    "reason": "biometric glitch",
                    "regularized_by": "hr.admin",
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["late_minutes"], 0)
            self.assertEqual(data["work_hours"], 9.03)
            self.assertEqual(len(data["history"]), 1)
            hist = data["history"][0]
            self.assertEqual(hist["by"], "hr.admin")
            self.assertEqual(hist["changes"]["late_minutes"]["to"], 0)
            self.assertEqual(hist["changes"]["punch_in"]["to"], int(new_in.timestamp() * 1000))


class TestAnalyticsEndpoints(unittest.TestCase):
    def test_employee_monthly(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find_one.return_value = {
                "emp_code": "EMP0001",
                "joined_on": "2026-01-05",
            }
            mock_db.attendance_logs.aggregate.return_value = [
                {
                    "present_days": 20.5,
                    "leave_days": 1,
                    "late_count": 2,
                    "total_late_minutes": 45,
                    "total_overtime_minutes": 40,
                }
            ]
            resp = test_client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["emp_code"], "EMP0001")
            self.assertEqual(data["month"], "2026-07")
            self.assertEqual(data["working_days"], 23)
            self.assertEqual(data["present_days"], 20.5)
            self.assertEqual(data["leave_days"], 1)
            self.assertEqual(data["late_count"], 2)
            self.assertEqual(data["total_late_minutes"], 45)
            self.assertEqual(data["total_overtime_minutes"], 40)
            self.assertEqual(data["attendance_pct"], 89.13)

    def test_department_summary_includes_zero_log_headcount(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.aggregate.return_value = [
                {
                    "department": "Engineering",
                    "headcount": 5,
                    "present_days": 18.0,
                    "avg_work_hours": 8.75,
                    "late_count": 1,
                    "total_late_minutes": 15,
                    "leave_count": 0,
                    "on_duty_count": 1,
                }
            ]
            resp = test_client.get("/analytics/departments/summary?month=2026-07&department=Engineering")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["month"], "2026-07")
            self.assertEqual(len(data["items"]), 1)
            item = data["items"][0]
            self.assertEqual(item["department"], "Engineering")
            self.assertEqual(item["headcount"], 5)
            self.assertEqual(item["avg_work_hours"], 8.75)

    def test_leaderboard_competition_ranking_and_limit(self):
        with patch("app.main.db") as mock_db:
            # Simulating ties: Rank 1, Rank 2, Rank 2 (tied), Rank 4
            mock_db.attendance_logs.aggregate.return_value = [
                {"rank": 1, "emp_code": "EMP0003", "name": "Rohan", "department": "Sales", "total_late_minutes": 80, "late_count": 3},
                {"rank": 2, "emp_code": "EMP0001", "name": "Asha", "department": "Sales", "total_late_minutes": 45, "late_count": 2},
                {"rank": 2, "emp_code": "EMP0004", "name": "Deepak", "department": "Sales", "total_late_minutes": 45, "late_count": 1},
            ]
            resp = test_client.get("/analytics/leaderboard/late?month=2026-07&limit=2")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["month"], "2026-07")
            # Limit is 2, but both tied at rank 2 are returned!
            self.assertEqual(len(data["items"]), 3)
            self.assertEqual(data["items"][0]["rank"], 1)
            self.assertEqual(data["items"][1]["rank"], 2)
            self.assertEqual(data["items"][2]["rank"], 2)

    def test_department_trend(self):
        with patch("app.main.db") as mock_db:
            mock_db.employees.find.return_value = [
                {"emp_code": "EMP0001", "joined_on": "2026-01-01"},
                {"emp_code": "EMP0002", "joined_on": "2026-01-01"},
            ]
            mock_db.attendance_logs.aggregate.return_value = [
                {
                    "date": "2026-07-06",
                    "is_working_day": True,
                    "headcount": 2,
                    "present_count": 2.0,
                    "late_count": 0,
                    "attendance_rate": 1.0000,
                    "moving_avg_7d": 1.0000,
                },
                {
                    "date": "2026-07-07",
                    "is_working_day": True,
                    "headcount": 2,
                    "present_count": 1.5,
                    "late_count": 1,
                    "attendance_rate": 0.7500,
                    "moving_avg_7d": 0.8750,
                },
            ]
            resp = test_client.get("/analytics/departments/Engineering/trend?from=2026-07-06&to=2026-07-07")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["department"], "Engineering")
            self.assertEqual(len(data["items"]), 2)
            self.assertEqual(data["items"][0]["date"], "2026-07-06")
            self.assertEqual(data["items"][0]["moving_avg_7d"], 1.0)
            self.assertEqual(data["items"][1]["moving_avg_7d"], 0.875)

    def test_explain_endpoint(self):
        with patch("app.main.db") as mock_db:
            mock_db.command.return_value = {"queryPlanner": {"winningPlan": {"stage": "IXSCAN"}}}
            resp = test_client.get("/admin/explain/attendance_list?page=1&page_size=20")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["endpoint"], "attendance_list")
            self.assertEqual(data["collection"], "attendance_logs")
            self.assertIn("queryPlanner", data["explain"])

            # Explain unknown endpoint -> 422
            resp_unknown = test_client.get("/admin/explain/unknown_endpoint")
            self.assertEqual(resp_unknown.status_code, 422)
