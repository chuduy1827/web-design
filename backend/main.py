import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware


logger = logging.getLogger(__name__)

app = FastAPI(title="Week 07-08 Lab - Items API + House Price Predictor")

# Week 08 — Sessions middleware. This stores a signed session cookie in the
# browser. Replace the development secret through SESSION_SECRET_KEY in real use.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "week-08-development-secret"),
    max_age=60 * 60,
    same_site="lax",
    https_only=False,
)

# In-memory storage for items

_items: dict[int, dict] = {}
_next_id: int = 1


# Week 08 — Middleware: attach a request ID, measure duration, and log every
# request after the response has been produced.
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "%s %s failed in %.2f ms request_id=%s",
            request.method,
            request.url.path,
            duration_ms,
            request_id,
        )
        raise

    duration_ms = (time.perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    logger.info(
        "%s %s -> %s in %.2f ms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


# Week 08 — Dependencies: one reusable dependency validates an item ID and
# returns the stored item to routes that need an existing item.
def get_existing_item(item_id: int) -> dict:
    item = _items.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


class SessionUser(BaseModel):
    username: str


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)


def get_current_user(request: Request) -> SessionUser:
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return SessionUser(username=username)


def _get_next_id() -> int:
    global _next_id
    nid = _next_id
    _next_id += 1
    return nid


class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Item name")
    price: float = Field(..., ge=0, description="Item price (non-negative)")


class ItemPublic(BaseModel):
    id: int
    name: str
    price: float


class ItemUpdate(BaseModel):
    """Part A — every field optional for PATCH."""
    name: str | None = Field(None, min_length=1)
    price: float | None = Field(None, ge=0)



class HousePriceRequest(BaseModel):
    area_sqm: float = Field(..., gt=0, description="Area in square meters, must be > 0")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms, must be >= 0")
    distance_to_center_km: float = Field(..., ge=0, description="Distance to city center in km")


class HousePricePrediction(BaseModel):
    predicted_price: float
    currency: str = "VND"


def predict_price(area: float, bedrooms: int, location: str) -> float:
    base_price = 500_000_000
    total = base_price + (15_000_000 * area) + (50_000_000 * bedrooms)

    loc = location.strip().lower()
    if loc == "hanoi":
        total *= 1.3
    elif loc == "hcmc":
        total *= 1.25

    return round(total / 1_000_000) * 1_000_000


@app.get("/predict")
def predict(area: float, bedrooms: int, location: str = "other"):
    price = predict_price(area, bedrooms, location)
    return {
        "area": area,
        "bedrooms": bedrooms,
        "location": location,
        "predicted_price": price,
    }


# Part E — POST /predict/house-price

@app.post("/predict/house-price", response_model=HousePricePrediction)
def predict_house_price(payload: HousePriceRequest):
    """
    Toy prediction endpoint.
    Formula: price = area_sqm * 15_000_000
                   - distance_to_center_km * 5_000_000
                   + bedrooms * 20_000_000
    Swapping this formula for model.predict(...) is a one-line change
    when a real ML model is plugged in.
    """
    price = (
        payload.area_sqm * 15_000_000
        - payload.distance_to_center_km * 5_000_000
        + payload.bedrooms * 20_000_000
    )
    return HousePricePrediction(predicted_price=float(price), currency="VND")




@app.get("/items", response_model=ItemListResponse)
def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    q: str | None = Query(None, min_length=2),
    sort_by: str = Query("id", pattern="^(id|name|price)$"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
):
    """
    Part B — filtering, searching & sorting.
    Order of operations: filter -> search -> sort -> paginate.
    Part D — returns envelope with total (after filtering, before slicing).
    """
    # Start from all items
    filtered = list(_items.values())

    # Filtering by price range
    if min_price is not None:
        filtered = [it for it in filtered if it["price"] >= min_price]
    if max_price is not None:
        filtered = [it for it in filtered if it["price"] <= max_price]

    # Case-insensitive substring search on name
    if q is not None:
        q_lower = q.lower()
        filtered = [it for it in filtered if q_lower in it["name"].lower()]

    # Sorting
    reverse = (order == "desc")
    if sort_by == "name":
        filtered.sort(key=lambda x: x["name"].lower(), reverse=reverse)
    elif sort_by == "price":
        filtered.sort(key=lambda x: x["price"], reverse=reverse)
    else:  # id
        filtered.sort(key=lambda x: x["id"], reverse=reverse)



@app.get("/items/{item_id}", response_model=ItemPublic)
def get_item(item: dict = Depends(get_existing_item)):
    return item


@app.put("/items/{item_id}", response_model=ItemPublic)
def update_item(
    payload: ItemCreate,
    existing: dict = Depends(get_existing_item),
):
    item_id = existing["id"]

    updated = {"id": item_id, "name": payload.name, "price": payload.price}
    _items[item_id] = updated
    return updated


@app.patch("/items/{item_id}", response_model=ItemPublic)
def patch_item(
    payload: ItemUpdate,
    existing: dict = Depends(get_existing_item),
):
    """Part A — partial update, only touch fields the client sent."""
    item_id = existing["id"]

    update_data = payload.model_dump(exclude_unset=True)


    for field, value in update_data.items():
        if value is not None:
            existing[field] = value

    _items[item_id] = existing
    return existing


@app.delete("/items/{item_id}", status_code=204)
def delete_item(existing: dict = Depends(get_existing_item)):
    del _items[existing["id"]]
    return None


def session_login(payload: LoginRequest, request: Request):
    request.session["username"] = payload.username.strip()
    return {"message": "Logged in", "username": request.session["username"]}


@app.get("/session/me")
def session_me(user: SessionUser = Depends(get_current_user)):
    return user


@app.post("/session/logout")
def session_logout(request: Request):
    request.session.clear()
    return {"message": "Logged out"}


_static_dir = Path(__file__).parent / ".." / "frontend"
_candidates = [
    Path(__file__).parent / "frontend",
    Path(__file__).parent / ".." / "frontend",
    Path("frontend"),
    Path("../frontend"),
]
for _c in _candidates:
    if _c.exists() and _c.is_dir():
        try:
            app.mount("/static", StaticFiles(directory=str(_c)), name="static")
            break
        except Exception:
            pass
