from fastapi import APIRouter
from datetime import datetime
from typing import List, Dict, Any

router = APIRouter()

@router.get("", tags=["diagnostics"])
async def get_diagnostics() -> Dict[str, Any]:
    """Return mock diagnostic metrics for model vs TIP comparison.

    The structure roughly matches what the front-end expects:
    - dual_cdf: arrays of {x, y}
    - box_per_inclination: list of {inclination, values}
    - dt_scatter: list of {x, y}
    - reliability: list of {p, observed}
    """
    dual_cdf = {
        "tip": [{"x": i * 0.1, "y": i / 50} for i in range(50)],
        "model": [{"x": i * 0.1, "y": min(1.0, (i / 50) ** 1.2)} for i in range(50)],
    }

    box_data = [
        {"inclination": band, "values": [v * 0.1 for v in [10, 15, 20, 25, 30]]}
        for band in ["0-10°", "10-30°", "30-60°", "60-90°"]
    ]

    scatter = [
        {"x": i * 0.3, "y": (i % 20) - 10}
        for i in range(100)
    ]

    reliability = [
        {"p": (i + 0.5) / 10, "observed": (i + 0.5) / 10 + (0.05 - 0.1 * ((i % 2)))}
        for i in range(10)
    ]

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dual_cdf": dual_cdf,
        "box_per_inclination": box_data,
        "dt_scatter": scatter,
        "reliability": reliability,
    } 