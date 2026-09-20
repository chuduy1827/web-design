from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path

app = FastAPI(title="Week 07 Extended Lab - Items API + House Price Predictor")

# In-memory storage for items

_items: dict[int, dict] = {}
_next_id: int = 1


def _get_next_id() -> int:
    global _next_id
    nid = _next_id
    _next_id += 1
    return nid


def _find_duplicate_name(name: str, exclude_id: int | None = None) -> bool:
    """Case-insensitive duplicate check."""
    lower = name.strip().lower()
    for iid, item in _items.items():
        if exclude_id is not None and iid == exclude_id:
            continue
        if item["name"].strip().lower() == lower:
            return True
    return False

# Pydantic models — Items

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


class ItemListResponse(BaseModel):
    """Part D — envelope pattern."""
    items: list[ItemPublic]
    total: int
    skip: int
    limit: int



# Pydantic models — House price prediction (Part E)

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


# Items CRUD

@app.post("/items", response_model=ItemPublic, status_code=201)
def create_item(payload: ItemCreate):
    # Part C — duplicate name check (case-insensitive)
    if _find_duplicate_name(payload.name):
        raise HTTPException(status_code=409, detail="Item with this name already exists")
    item_id = _get_next_id()
    item = {"id": item_id, "name": payload.name, "price": payload.price}
    _items[item_id] = item
    return item


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

    total = len(filtered)

    # Pagination (must be last)
    paginated = filtered[skip: skip + limit]

    return ItemListResponse(
        items=[ItemPublic(**it) for it in paginated],
        total=total,
        skip=skip,
        limit=limit,
    )


@app.get("/items/{item_id}", response_model=ItemPublic)
def get_item(item_id: int):
    item = _items.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.put("/items/{item_id}", response_model=ItemPublic)
def update_item(item_id: int, payload: ItemCreate):
    """Full replace — PUT semantics."""
    existing = _items.get(item_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Item not found")

    # Part C — duplicate check when renaming 
    if payload.name.strip().lower() != existing["name"].strip().lower():
        if _find_duplicate_name(payload.name, exclude_id=item_id):
            raise HTTPException(status_code=409, detail="Item with this name already exists")

    updated = {"id": item_id, "name": payload.name, "price": payload.price}
    _items[item_id] = updated
    return updated


@app.patch("/items/{item_id}", response_model=ItemPublic)
def patch_item(item_id: int, payload: ItemUpdate):
    """Part A — partial update, only touch fields the client sent."""
    existing = _items.get(item_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Item not found")

    update_data = payload.model_dump(exclude_unset=True)

    # Part C — if name is being changed, check for duplicates
    if "name" in update_data and update_data["name"] is not None:
        new_name = update_data["name"]
        if new_name.strip().lower() != existing["name"].strip().lower():
            if _find_duplicate_name(new_name, exclude_id=item_id):
                raise HTTPException(status_code=409, detail="Item with this name already exists")

    # Only update provided fields
    for field, value in update_data.items():
        if value is not None:
            existing[field] = value

    _items[item_id] = existing
    return existing


@app.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int):
    if item_id not in _items:
        raise HTTPException(status_code=404, detail="Item not found")
    del _items[item_id]
    return None

_static_dir = Path(__file__).parent / ".." / "frontend"
# Try common locations
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
