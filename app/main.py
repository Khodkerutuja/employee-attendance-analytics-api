"""
Employee Attendance & Analytics API - Production Implementation

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""
import calendar
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import os
from typing import Any, Dict, List, Optional

from bson import SON
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), tz_aware=True)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))


def init_indexes():
    """Create all required MongoDB indexes idempotently at startup."""
    # 1. Employees: uniqueness and query coverage
    db.employees.create_index([("emp_code", 1)], unique=True)
    db.employees.create_index([("department", 1), ("emp_code", 1)])
    db.employees.create_index([("department", 1), ("joined_on", 1)])
    db.employees.create_index([("joined_on", 1)])

    # 2. Attendance logs: natural key uniqueness, listing, and analytics indexing
    db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True)
    db.attendance_logs.create_index([("date", -1), ("emp_code", 1)])
    db.attendance_logs.create_index([("emp_code", 1), ("punch_in", -1)])
    db.attendance_logs.create_index([("date", 1), ("late_minutes", 1)])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_indexes()
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def epoch_ms_to_datetime(ms: int) -> datetime:
    """Truncates milliseconds to whole seconds and returns UTC datetime (R1)."""
    sec = ms // 1000
    return datetime.fromtimestamp(sec, tz=timezone.utc)


def datetime_to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    """Converts a BSON or Python datetime to epoch milliseconds."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def round_2dp(val: Optional[float | Decimal]) -> Optional[float]:
    """Rounds to 2 decimals half-up (R8)."""
    if val is None:
        return None
    d = Decimal(str(val))
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def round_4dp(val: Optional[float | Decimal]) -> Optional[float]:
    """Rounds to 4 decimals half-up (R8)."""
    if val is None:
        return None
    d = Decimal(str(val))
    return float(d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def count_working_days(month_str: str, joined_on_str: str) -> int:
    """
    R7: Working days Mon-Fri from joined_on for mid-month joiners within the month.
    """
    y, m = map(int, month_str.split("-"))
    num_days = calendar.monthrange(y, m)[1]
    start_date = max(date(y, m, 1), date.fromisoformat(joined_on_str))
    end_date = date(y, m, num_days)
    if start_date > end_date:
        return 0
    working_days = 0
    curr = start_date
    while curr <= end_date:
        if curr.weekday() < 5:  # Monday=0, ..., Friday=4
            working_days += 1
        curr += timedelta(days=1)
    return working_days


def compute_attendance_date(punch_in_ist: datetime, shift_start: str, shift_end: str) -> str:
    """
    R1: Shift times and attendance dates are IST. A record's date is the IST calendar
    date of its punch_in. Overnight shifts (shift_end <= shift_start): punch-in earlier
    than shift_end belongs to the shift that started the previous day.
    """
    is_overnight = shift_end <= shift_start
    if is_overnight:
        end_h, end_m = map(int, shift_end.split(":"))
        if punch_in_ist.time() < time(end_h, end_m):
            return (punch_in_ist.date() - timedelta(days=1)).isoformat()
    return punch_in_ist.date().isoformat()


def compute_late_minutes(
    punch_in: datetime, shift_start: str, date_str: Optional[str] = None
) -> int:
    """
    R1, R2: Grace period 10 minutes. Late only if punch-in is strictly more than 10:00 (600s)
    after shift_start; then late_minutes = whole minutes (floor) since shift_start.
    Instants are truncated to whole seconds before calculation.
    """
    if punch_in.tzinfo is None:
        punch_in = punch_in.replace(tzinfo=timezone.utc)
    punch_in_ist = punch_in.astimezone(IST).replace(microsecond=0)

    start_h, start_m = map(int, shift_start.split(":"))
    if date_str:
        start_dt = datetime.fromisoformat(date_str).replace(
            hour=start_h, minute=start_m, second=0, microsecond=0, tzinfo=IST
        )
    else:
        start_dt = punch_in_ist.replace(
            hour=start_h, minute=start_m, second=0, microsecond=0
        )
        if (punch_in_ist - start_dt).total_seconds() < -12 * 3600:
            start_dt -= timedelta(days=1)

    diff_sec = int((punch_in_ist - start_dt).total_seconds())
    if diff_sec > 600:
        return diff_sec // 60
    return 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """
    R4: (punch_out - punch_in) in seconds / 3600, rounded to 2 decimals half-up
    (not banker's rounding, not float rounding). Instants truncated to whole seconds.
    """
    if punch_in.tzinfo is None:
        punch_in = punch_in.replace(tzinfo=timezone.utc)
    if punch_out.tzinfo is None:
        punch_out = punch_out.replace(tzinfo=timezone.utc)
    in_sec = int(punch_in.timestamp())
    out_sec = int(punch_out.timestamp())
    seconds = out_sec - in_sec
    if seconds <= 0:
        return 0.0
    val = Decimal(seconds) / Decimal(3600)
    return float(val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(
    punch_out: datetime, shift_end: str, date_str: str, shift_start: Optional[str] = None
) -> int:
    """
    R3: Whole minutes (floor) from shift_end to punch_out, counted only if at least 30;
    otherwise 0. For overnight shifts (shift_end <= shift_start), shift ends next calendar day.
    """
    if punch_out.tzinfo is None:
        punch_out = punch_out.replace(tzinfo=timezone.utc)
    punch_out_ist = punch_out.astimezone(IST).replace(microsecond=0)

    end_h, end_m = map(int, shift_end.split(":"))
    end_dt = datetime.fromisoformat(date_str).replace(
        hour=end_h, minute=end_m, second=0, microsecond=0, tzinfo=IST
    )
    is_overnight = (shift_start is not None and shift_end <= shift_start) or (
        shift_start is None and (punch_out_ist - end_dt).total_seconds() > 18 * 3600
    )
    if is_overnight:
        end_dt += timedelta(days=1)

    diff_sec = int((punch_out_ist - end_dt).total_seconds())
    if diff_sec <= 0:
        return 0
    minutes = diff_sec // 60
    return minutes if minutes >= 30 else 0


def compute_half_day(work_hours: Optional[float]) -> bool:
    """
    R5: After punch-out, half_day is true when the rounded work_hours < 4.50.
    """
    if work_hours is None:
        return False
    return work_hours < 4.50


def format_history_entry(entry: dict) -> dict:
    at_val = entry.get("at")
    if isinstance(at_val, datetime):
        at_val = datetime_to_epoch_ms(at_val)
    formatted = {
        "at": at_val,
        "by": entry.get("by"),
        "reason": entry.get("reason"),
        "changes": {},
    }
    for field, change in entry.get("changes", {}).items():
        c_from = change.get("from")
        c_to = change.get("to")
        if isinstance(c_from, datetime):
            c_from = datetime_to_epoch_ms(c_from)
        if isinstance(c_to, datetime):
            c_to = datetime_to_epoch_ms(c_to)
        formatted["changes"][field] = {"from": c_from, "to": c_to}
    return formatted


def format_employee(doc: dict) -> dict:
    doc.pop("_id", None)
    doc.pop("id", None)
    created_at = doc.get("created_at")
    if isinstance(created_at, datetime):
        created_at = datetime_to_epoch_ms(created_at)
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc.get("shift_start", "09:30"),
        "shift_end": doc.get("shift_end", "18:30"),
        "joined_on": doc["joined_on"],
        "created_at": created_at,
    }


def format_attendance_record(doc: dict) -> dict:
    doc.pop("_id", None)
    doc.pop("id", None)
    punch_in = doc.get("punch_in")
    if isinstance(punch_in, datetime):
        punch_in = datetime_to_epoch_ms(punch_in)
    punch_out = doc.get("punch_out")
    if isinstance(punch_out, datetime):
        punch_out = datetime_to_epoch_ms(punch_out)
    history = [format_history_entry(h) for h in doc.get("history", [])]
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": punch_in,
        "punch_out": punch_out,
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class StatusEnum(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class PresenceStatusEnum(str, Enum):
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def validate_shift_and_date(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        try:
            datetime.strptime(self.joined_on, "%Y-%m-%d")
        except ValueError:
            raise ValueError("joined_on must be a valid date in YYYY-MM-DD format")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(
        default=None,
        ge=100_000_000_000,
        le=4_102_444_800_000,
        strict=True,
    )
    status: PresenceStatusEnum = PresenceStatusEnum.PRESENT


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(
        default=None,
        ge=100_000_000_000,
        le=4_102_444_800_000,
        strict=True,
    )


class RegularizeIn(BaseModel):
    status: Optional[StatusEnum] = None
    punch_in: Optional[int] = Field(
        default=None,
        ge=100_000_000_000,
        le=4_102_444_800_000,
        strict=True,
    )
    punch_out: Optional[int] = Field(
        default=None,
        ge=100_000_000_000,
        le=4_102_444_800_000,
        strict=True,
    )
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    now_sec = int(datetime.now(timezone.utc).timestamp())
    now_dt = datetime.fromtimestamp(now_sec, tz=timezone.utc)
    doc = body.model_dump()
    doc["created_at"] = now_dt
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")
    return format_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {}
    if department:
        q["department"] = department
    skip = (page - 1) * page_size
    total = db.employees.count_documents(q)
    cursor = db.employees.find(q).sort("emp_code", 1).skip(skip).limit(page_size)
    items = [format_employee(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="unknown employee")

    if body.punched_at is not None:
        ts_utc = epoch_ms_to_datetime(body.punched_at)
    else:
        now_sec = int(datetime.now(timezone.utc).timestamp())
        ts_utc = datetime.fromtimestamp(now_sec, tz=timezone.utc)

    ts_ist = ts_utc.astimezone(IST)
    shift_start = emp.get("shift_start", "09:30")
    shift_end = emp.get("shift_end", "18:30")
    d = compute_attendance_date(ts_ist, shift_start, shift_end)
    late_minutes = compute_late_minutes(ts_ist, shift_start, d)

    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status.value,
        "punch_in": ts_utc,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="already punched in for this date")

    return format_attendance_record(doc)


@app.post("/attendance/punch-out", status_code=200)
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="unknown employee")

    if body.punched_at is not None:
        ts_utc = epoch_ms_to_datetime(body.punched_at)
    else:
        now_sec = int(datetime.now(timezone.utc).timestamp())
        ts_utc = datetime.fromtimestamp(now_sec, tz=timezone.utc)

    # Closes the employee's most recent record whose punch_in is at or before punched_at
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts_utc}},
        sort=[("punch_in", -1)],
    )
    if not record:
        raise HTTPException(status_code=404, detail="no punch-in found")

    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="record is already punched out")

    punch_in_dt = record["punch_in"]
    if punch_in_dt.tzinfo is None:
        punch_in_dt = punch_in_dt.replace(tzinfo=timezone.utc)

    if ts_utc <= punch_in_dt:
        raise HTTPException(status_code=422, detail="punched_at must be after punch_in")

    if ts_utc > punch_in_dt + timedelta(hours=24):
        raise HTTPException(status_code=422, detail="punched_at cannot be more than 24 hours after punch_in")

    work_hours = compute_work_hours(punch_in_dt, ts_utc)
    shift_end = emp.get("shift_end", "18:30")
    shift_start = emp.get("shift_start", "09:30")
    overtime = compute_overtime(ts_utc, shift_end, record["date"], shift_start)
    half_day = compute_half_day(work_hours)

    updated_doc = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": ts_utc,
                "work_hours": work_hours,
                "overtime_minutes": overtime,
                "half_day": half_day,
            }
        },
        return_document=pymongo.ReturnDocument.AFTER,
    )
    if not updated_doc:
        raise HTTPException(status_code=409, detail="record is already punched out")

    return format_attendance_record(updated_doc)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[StatusEnum] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be <= date_to")

    q = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status.value

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(q)
        .sort([("date", -1), ("emp_code", 1)])
        .skip(skip)
        .limit(page_size)
    )
    items = [format_attendance_record(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}", status_code=200)
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="unknown employee")

    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(status_code=404, detail="no record found for this date")

    # Rules:
    # Setting status ABSENT or LEAVE clears both punch times, so supplying punch_in or punch_out with it is 422
    if body.status in (StatusEnum.ABSENT, StatusEnum.LEAVE):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                status_code=422,
                detail="punch times cannot be supplied when setting status to ABSENT or LEAVE",
            )
        final_status = body.status.value
        final_punch_in = None
        final_punch_out = None
    elif body.status is None and record.get("status") in ("ABSENT", "LEAVE") and (body.punch_in is not None or body.punch_out is not None):
        raise HTTPException(
            status_code=422,
            detail="cannot set punch times on ABSENT or LEAVE record without changing status",
        )
    else:
        final_status = body.status.value if body.status is not None else record["status"]
        if body.punch_in is not None:
            final_punch_in = epoch_ms_to_datetime(body.punch_in)
        else:
            final_punch_in = record.get("punch_in")
            if isinstance(final_punch_in, datetime) and final_punch_in.tzinfo is None:
                final_punch_in = final_punch_in.replace(tzinfo=timezone.utc)

        # Presence status requires punch_in (supplied or stored)
        if final_status in ("PRESENT", "WFH", "ON_DUTY") and not final_punch_in:
            raise HTTPException(status_code=422, detail="presence status requires punch_in")

        # punch_in must stay on the record's attendance date (R1)
        if final_punch_in is not None:
            punch_in_ist = final_punch_in.astimezone(IST)
            shift_start = emp.get("shift_start", "09:30")
            shift_end = emp.get("shift_end", "18:30")
            att_date = compute_attendance_date(punch_in_ist, shift_start, shift_end)
            if att_date != date:
                raise HTTPException(status_code=422, detail="punch_in must stay on the record's attendance date")

        if body.punch_out is not None:
            final_punch_out = epoch_ms_to_datetime(body.punch_out)
        else:
            final_punch_out = record.get("punch_out")
            if isinstance(final_punch_out, datetime) and final_punch_out.tzinfo is None:
                final_punch_out = final_punch_out.replace(tzinfo=timezone.utc)

        if final_punch_out is not None:
            if final_punch_in is None:
                raise HTTPException(status_code=422, detail="punch_out requires punch_in")
            if final_punch_out <= final_punch_in:
                raise HTTPException(status_code=422, detail="punch_out must be after punch_in")
            if final_punch_out > final_punch_in + timedelta(hours=24):
                raise HTTPException(status_code=422, detail="punch_out must be within 24 hours of punch_in")

    # Recompute derived fields
    if final_status in ("ABSENT", "LEAVE"):
        new_late_minutes = 0
        new_work_hours = None
        new_overtime_minutes = 0
        new_half_day = False
    else:
        new_late_minutes = compute_late_minutes(final_punch_in, emp.get("shift_start", "09:30"), date)
        if final_punch_out is not None:
            new_work_hours = compute_work_hours(final_punch_in, final_punch_out)
            new_overtime_minutes = compute_overtime(
                final_punch_out, emp.get("shift_end", "18:30"), date, emp.get("shift_start", "09:30")
            )
            new_half_day = compute_half_day(new_work_hours)
        else:
            new_work_hours = None
            new_overtime_minutes = 0
            new_half_day = False

    # Compute changes
    old_punch_in = record.get("punch_in")
    if isinstance(old_punch_in, datetime) and old_punch_in.tzinfo is None:
        old_punch_in = old_punch_in.replace(tzinfo=timezone.utc)
    old_punch_out = record.get("punch_out")
    if isinstance(old_punch_out, datetime) and old_punch_out.tzinfo is None:
        old_punch_out = old_punch_out.replace(tzinfo=timezone.utc)

    changes = {}
    if record.get("status") != final_status:
        changes["status"] = {"from": record.get("status"), "to": final_status}
    if old_punch_in != final_punch_in:
        changes["punch_in"] = {"from": old_punch_in, "to": final_punch_in}
    if old_punch_out != final_punch_out:
        changes["punch_out"] = {"from": old_punch_out, "to": final_punch_out}
    if record.get("work_hours") != new_work_hours:
        changes["work_hours"] = {"from": record.get("work_hours"), "to": new_work_hours}
    if record.get("late_minutes", 0) != new_late_minutes:
        changes["late_minutes"] = {"from": record.get("late_minutes", 0), "to": new_late_minutes}
    if record.get("overtime_minutes", 0) != new_overtime_minutes:
        changes["overtime_minutes"] = {"from": record.get("overtime_minutes", 0), "to": new_overtime_minutes}
    if record.get("half_day", False) != new_half_day:
        changes["half_day"] = {"from": record.get("half_day", False), "to": new_half_day}

    if not changes:
        raise HTTPException(status_code=422, detail="a request that changes nothing is 422")

    now_sec = int(datetime.now(timezone.utc).timestamp())
    now_dt = datetime.fromtimestamp(now_sec, tz=timezone.utc)
    history_entry = {
        "at": now_dt,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    curr_hist = record.get("history", [])
    filter_q = (
        {"_id": record["_id"], "history": {"$size": len(curr_hist)}}
        if "history" in record
        else {"_id": record["_id"]}
    )

    updated_doc = db.attendance_logs.find_one_and_update(
        filter_q,
        {
            "$set": {
                "status": final_status,
                "punch_in": final_punch_in,
                "punch_out": final_punch_out,
                "work_hours": new_work_hours,
                "late_minutes": new_late_minutes,
                "overtime_minutes": new_overtime_minutes,
                "half_day": new_half_day,
            },
            "$push": {"history": history_entry},
        },
        return_document=pymongo.ReturnDocument.AFTER,
    )
    if not updated_doc:
        raise HTTPException(status_code=409, detail="concurrent regularization conflict")

    return format_attendance_record(updated_doc)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="unknown employee")

    y, m = map(int, month.split("-"))
    last_day = calendar.monthrange(y, m)[1]
    working_days = count_working_days(month, emp.get("joined_on", "1970-01-01"))

    pipeline = [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": f"{month}-01", "$lte": f"{month}-{last_day:02d}"},
            }
        },
        {
            "$project": {
                "status": 1,
                "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                "half_day": {"$ifNull": ["$half_day", False]},
                "dayOfWeek": {
                    "$dayOfWeek": {
                        "$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}
                    }
                },
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$in": ["$dayOfWeek", [2, 3, 4, 5, 6]]},
                                ]
                            },
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"},
            }
        },
    ]

    res = list(db.attendance_logs.aggregate(pipeline))
    if res:
        data = res[0]
        present_days = float(data.get("present_days", 0.0))
        leave_days = int(data.get("leave_days", 0))
        late_count = int(data.get("late_count", 0))
        total_late = int(data.get("total_late_minutes", 0))
        total_ot = int(data.get("total_overtime_minutes", 0))
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late = 0
        total_ot = 0

    if working_days == 0:
        attendance_pct = None
    else:
        pct = (Decimal(str(present_days)) / Decimal(working_days)) * Decimal(100)
        attendance_pct = float(pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_ot,
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    y, m = map(int, month.split("-"))
    last_day = calendar.monthrange(y, m)[1]
    month_end = f"{month}-{last_day:02d}"
    month_start = f"{month}-01"

    match_emp = {"joined_on": {"$lte": month_end}}
    if department:
        match_emp["department"] = department

    pipeline = [
        {"$match": match_emp},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp_code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp_code"]},
                                    {"$gte": ["$date", month_start]},
                                    {"$lte": ["$date", month_end]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "status": 1,
                            "work_hours": 1,
                            "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                            "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                            "half_day": {"$ifNull": ["$half_day", False]},
                            "dayOfWeek": {
                                "$dayOfWeek": {
                                    "$dateFromString": {
                                        "dateString": "$date",
                                        "format": "%Y-%m-%d",
                                    }
                                }
                            },
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "all_logs": {
                    "$push": "$logs"
                },
            }
        },
        {
            "$project": {
                "department": "$_id",
                "headcount": 1,
                "flat_logs": {
                    "$reduce": {
                        "input": "$all_logs",
                        "initialValue": [],
                        "in": {"$concatArrays": ["$$value", "$$this"]},
                    }
                },
            }
        },
        {
            "$project": {
                "department": 1,
                "headcount": 1,
                "present_days": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$flat_logs",
                                    "as": "log",
                                    "cond": {
                                        "$and": [
                                            {"$in": ["$$log.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                            {"$in": ["$$log.dayOfWeek", [2, 3, 4, 5, 6]]},
                                        ]
                                    },
                                }
                            },
                            "as": "plog",
                            "in": {"$cond": ["$$plog.half_day", 0.5, 1.0]},
                        }
                    }
                },
                "avg_work_hours": {
                    "$avg": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$flat_logs",
                                    "as": "log",
                                    "cond": {
                                        "$and": [
                                            {"$in": ["$$log.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                            {"$ne": ["$$log.work_hours", None]},
                                        ]
                                    },
                                }
                            },
                            "as": "wlog",
                            "in": "$$wlog.work_hours",
                        }
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$flat_logs",
                            "as": "log",
                            "cond": {"$gt": ["$$log.late_minutes", 0]},
                        }
                    }
                },
                "total_late_minutes": {
                    "$sum": {
                        "$map": {
                            "input": "$flat_logs",
                            "as": "log",
                            "in": "$$log.late_minutes",
                        }
                    }
                },
                "leave_count": {
                    "$size": {
                        "$filter": {
                            "input": "$flat_logs",
                            "as": "log",
                            "cond": {"$eq": ["$$log.status", "LEAVE"]},
                        }
                    }
                },
                "on_duty_count": {
                    "$size": {
                        "$filter": {
                            "input": "$flat_logs",
                            "as": "log",
                            "cond": {"$eq": ["$$log.status", "ON_DUTY"]},
                        }
                    }
                },
            }
        },
        {"$sort": {"department": 1}},
    ]

    docs = list(db.employees.aggregate(pipeline))
    items = []
    for d in docs:
        avg_wh = d.get("avg_work_hours")
        if avg_wh is not None:
            avg_wh = round_2dp(avg_wh)
        items.append({
            "department": d["department"],
            "headcount": d["headcount"],
            "present_days": float(d.get("present_days", 0.0)),
            "avg_work_hours": avg_wh,
            "late_count": int(d.get("late_count", 0)),
            "total_late_minutes": int(d.get("total_late_minutes", 0)),
            "leave_count": int(d.get("leave_count", 0)),
            "on_duty_count": int(d.get("on_duty_count", 0)),
        })

    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    y, m = map(int, month.split("-"))
    last_day = calendar.monthrange(y, m)[1]
    month_end = f"{month}-{last_day:02d}"
    month_start = f"{month}-01"

    pipeline = [
        {
            "$match": {
                "date": {"$gte": month_start, "$lte": month_end},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.extend([
        {
            "$project": {
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {
                    "rank": {"$rank": {}}
                },
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}},
    ])

    docs = list(db.attendance_logs.aggregate(pipeline))
    items = []
    for d in docs:
        items.append({
            "rank": d["rank"],
            "emp_code": d["emp_code"],
            "name": d["name"],
            "department": d["department"],
            "total_late_minutes": d["total_late_minutes"],
            "late_count": d["late_count"],
        })

    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: str = Query(..., alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: str = Query(..., alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    try:
        d_from = date.fromisoformat(from_date)
        d_to = date.fromisoformat(to_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid date format")

    if d_to < d_from:
        raise HTTPException(status_code=422, detail="to must be >= from")

    num_days = (d_to - d_from).days + 1
    if num_days > 92:
        raise HTTPException(status_code=422, detail="date range cannot exceed 92 days")

    # Check department exists in employees collection
    dept_emps = list(db.employees.find({"department": department}, {"emp_code": 1, "joined_on": 1}))
    if not dept_emps:
        raise HTTPException(status_code=404, detail="unknown department")

    emp_codes = [e["emp_code"] for e in dept_emps]

    to_dt_plus_1 = datetime.fromisoformat(to_date) + timedelta(days=1)
    pipeline = [
        {
            "$match": {
                "emp_code": {"$in": emp_codes},
                "date": {"$gte": from_date, "$lte": to_date},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "date_str": "$_id",
                "date_dt": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}},
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$densify": {
                "field": "date_dt",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [datetime.fromisoformat(from_date), to_dt_plus_1],
                },
            }
        },
        {
            "$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date_dt"}},
                "dayOfWeek": {"$dayOfWeek": "$date_dt"},
                "present_count": {"$ifNull": ["$present_count", 0.0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"current_date": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$current_date"]},
                                ]
                            }
                        }
                    },
                    {"$count": "count"},
                ],
                "as": "hc",
            }
        },
        {
            "$addFields": {
                "headcount": {"$ifNull": [{"$arrayElemAt": ["$hc.count", 0]}, 0]},
                "is_working_day": {"$in": ["$dayOfWeek", [2, 3, 4, 5, 6]]},
            }
        },
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                {"$eq": ["$is_working_day", True]},
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {"$divide": ["$present_count", "$headcount"]},
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {
                            "documents": [-6, "current"]
                        }
                    }
                }
            }
        },
        {"$sort": {"date": 1}},
    ]

    docs = list(db.attendance_logs.aggregate(pipeline))

    # If attendance_logs had no records in the range, densify produces 0 documents.
    # Fallback to programmatic calendar filling:
    if len(docs) < num_days:
        existing_by_date = {d["date"]: d for d in docs}
        filled_docs = []
        curr = d_from
        while curr <= d_to:
            curr_str = curr.isoformat()
            if curr_str in existing_by_date:
                filled_docs.append(existing_by_date[curr_str])
            else:
                is_wd = curr.weekday() < 5
                hc = sum(1 for e in dept_emps if e.get("joined_on", "") <= curr_str)
                att_rate = 0.0 if (is_wd and hc > 0) else None
                filled_docs.append({
                    "date": curr_str,
                    "is_working_day": is_wd,
                    "headcount": hc,
                    "present_count": 0.0,
                    "late_count": 0,
                    "attendance_rate": att_rate,
                    "moving_avg_7d": att_rate,
                })
            curr += timedelta(days=1)
        docs = filled_docs

    items = []
    for d in docs:
        rate = d.get("attendance_rate")
        mavg = d.get("moving_avg_7d")
        items.append({
            "date": d["date"],
            "is_working_day": d["is_working_day"],
            "headcount": d["headcount"],
            "present_count": float(d.get("present_count", 0.0)),
            "late_count": int(d.get("late_count", 0)),
            "attendance_rate": round_4dp(rate) if rate is not None else None,
            "moving_avg_7d": round_4dp(mavg) if mavg is not None else None,
        })

    return {"department": department, "items": items}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[StatusEnum] = None,
    from_date: Optional[str] = Query(None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: Optional[str] = Query(None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must be <= date_to")
        q = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status.value
        skip = (page - 1) * page_size
        cmd = {
            "find": "attendance_logs",
            "filter": q,
            "sort": SON([("date", -1), ("emp_code", 1)]),
            "skip": skip,
            "limit": page_size,
        }
        res = db.command("explain", cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": res}

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(status_code=422, detail="emp_code and month are required")
        y, m = map(int, month.split("-"))
        last_day = calendar.monthrange(y, m)[1]
        pipeline = [
            {
                "$match": {
                    "emp_code": emp_code,
                    "date": {"$gte": f"{month}-01", "$lte": f"{month}-{last_day:02d}"},
                }
            },
            {
                "$project": {
                    "status": 1,
                    "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                    "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                    "half_day": {"$ifNull": ["$half_day", False]},
                    "dayOfWeek": {
                        "$dayOfWeek": {
                            "$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}
                        }
                    },
                }
            },
            {
                "$group": {
                    "_id": None,
                    "present_days": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                        {"$in": ["$dayOfWeek", [2, 3, 4, 5, 6]]},
                                    ]
                                },
                                {"$cond": ["$half_day", 0.5, 1.0]},
                                0.0,
                            ]
                        }
                    },
                    "leave_days": {
                        "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                    },
                    "late_count": {
                        "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                    },
                    "total_late_minutes": {"$sum": "$late_minutes"},
                    "total_overtime_minutes": {"$sum": "$overtime_minutes"},
                }
            },
        ]
        cmd = {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}}
        res = db.command("explain", cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": res}

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(status_code=422, detail="month is required")
        y, m = map(int, month.split("-"))
        last_day = calendar.monthrange(y, m)[1]
        month_end = f"{month}-{last_day:02d}"
        month_start = f"{month}-01"
        match_emp = {"joined_on": {"$lte": month_end}}
        if department:
            match_emp["department"] = department
        pipeline = [
            {"$match": match_emp},
            {
                "$lookup": {
                    "from": "attendance_logs",
                    "let": {"emp_code": "$emp_code"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$emp_code", "$$emp_code"]},
                                        {"$gte": ["$date", month_start]},
                                        {"$lte": ["$date", month_end]},
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "logs",
                }
            },
            {"$group": {"_id": "$department", "headcount": {"$sum": 1}}},
        ]
        cmd = {"aggregate": "employees", "pipeline": pipeline, "cursor": {}}
        res = db.command("explain", cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "employees", "explain": res}

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(status_code=422, detail="month is required")
        y, m = map(int, month.split("-"))
        last_day = calendar.monthrange(y, m)[1]
        month_end = f"{month}-{last_day:02d}"
        month_start = f"{month}-01"
        pipeline = [
            {
                "$match": {
                    "date": {"$gte": month_start, "$lte": month_end},
                    "late_minutes": {"$gt": 0},
                }
            },
            {
                "$group": {
                    "_id": "$emp_code",
                    "total_late_minutes": {"$sum": "$late_minutes"},
                    "late_count": {"$sum": 1},
                }
            },
        ]
        cmd = {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}}
        res = db.command("explain", cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": res}

    elif endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(status_code=422, detail="department, from, and to are required")
        if to_date < from_date:
            raise HTTPException(status_code=422, detail="to must be >= from")
        num_days = (date.fromisoformat(to_date) - date.fromisoformat(from_date)).days + 1
        if num_days > 92:
            raise HTTPException(status_code=422, detail="date range cannot exceed 92 days")

        dept_emps = list(db.employees.find({"department": department}, {"emp_code": 1}))
        emp_codes = [e["emp_code"] for e in dept_emps]
        pipeline = [
            {
                "$match": {
                    "emp_code": {"$in": emp_codes},
                    "date": {"$gte": from_date, "$lte": to_date},
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "present_count": {"$sum": 1},
                }
            },
        ]
        cmd = {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}}
        res = db.command("explain", cmd, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": res}

    else:
        raise HTTPException(status_code=422, detail=f"unknown endpoint '{endpoint}'")
