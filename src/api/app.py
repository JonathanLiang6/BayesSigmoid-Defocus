from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import DATA_DIR, RESULTS_DIR, FEATURE_CANDIDATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Defocus Dose Bayesian Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLEANED_DATA_PATH = DATA_DIR / "results" / "cleaned_data.csv"
LEARNING_SESSIONS_PATH = RESULTS_DIR / "learning_sessions.json"

learning_sessions: Dict[str, Dict[str, Any]] = {}


def load_learning_sessions() -> Dict[str, Dict[str, Any]]:
    if LEARNING_SESSIONS_PATH.exists():
        with open(LEARNING_SESSIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["patient_id"]: item for item in data}
    return {}


learning_cache = load_learning_sessions()


def get_patients_data() -> pd.DataFrame:
    if not CLEANED_DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Data not found, please run analysis first")
    return pd.read_csv(CLEANED_DATA_PATH)


@app.get("/api/patients")
async def list_patients():
    df = get_patients_data()
    patients = []
    for _, row in df.iterrows():
        patients.append({
            "id": f"{int(row['被测者编号'])}_{row['眼别']}",
            "subject_id": int(row['被测者编号']),
            "eye": row['眼别'],
            "batch": row['检测批次'],
            "age": float(row['年龄 (岁)']),
            "gender": int(row['性别']),
            "axial_length": float(row['眼轴长度 (mm)']),
            "choroid_thickness": float(row['黄斑中心凹下脉络膜厚度 (μm)']),
            "cvi": float(row['脉络膜血管指数 (CVI)']),
            "rdv15": float(row['RDV15(D)']),
            "refractive_error": float(row['验光 - 离焦量 (近视度数，D)']),
            "dose": float(row['离焦剂量 (D)']),
        })
    return {"patients": patients}


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    df = get_patients_data()
    
    parts = patient_id.split("_")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")
    
    subject_id = int(parts[0])
    eye = parts[1]
    
    row = df[(df['被测者编号'] == subject_id) & (df['眼别'] == eye)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    row = row.iloc[0]
    return {
        "id": patient_id,
        "subject_id": int(row['被测者编号']),
        "eye": row['眼别'],
        "batch": row['检测批次'],
        "age": float(row['年龄 (岁)']),
        "gender": int(row['性别']),
        "axial_length": float(row['眼轴长度 (mm)']),
        "choroid_thickness": float(row['黄斑中心凹下脉络膜厚度 (μm)']),
        "cvi": float(row['脉络膜血管指数 (CVI)']),
        "rdv15": float(row['RDV15(D)']),
        "refractive_error": float(row['验光 - 离焦量 (近视度数，D)']),
        "dose": float(row['离焦剂量 (D)']),
    }


@app.post("/api/learning/run/{patient_id}")
async def run_learning(patient_id: str):
    global learning_cache
    
    df = get_patients_data()
    
    parts = patient_id.split("_")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")
    
    subject_id = int(parts[0])
    eye = parts[1]
    
    row = df[(df['被测者编号'] == subject_id) & (df['眼别'] == eye)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if patient_id in learning_cache:
        data = learning_cache[patient_id]
    else:
        learning_cache = load_learning_sessions()
        if patient_id not in learning_cache:
            raise HTTPException(status_code=500, detail="Learning session not precomputed, please run analysis first")
        data = learning_cache[patient_id]
    
    session_id = str(uuid.uuid4())
    learning_sessions[session_id] = data
    
    logger.info(f"Learning session retrieved: {session_id}, patient: {patient_id}, rounds: {len(data['iterations'])}")
    
    return {
        "session_id": session_id,
        "patient_id": patient_id,
        "iterations": data["iterations"],
        "uncertainties": data["uncertainties"],
        "recommended_doses": data["recommended_doses"],
        "predicted_responses": data["predicted_responses"],
        "dose_response_curves": data["dose_response_curves"],
        "final_dose": data["final_dose"],
        "final_response": data["final_response"],
        "final_uncertainty": data["final_uncertainty"],
        "confidence": data["confidence"],
        "converged": data["converged"],
    }


@app.get("/api/learning/{session_id}")
async def get_learning_session(session_id: str):
    if session_id not in learning_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return learning_sessions[session_id]


@app.get("/api/learning/precache")
async def get_precache_status():
    global learning_cache
    learning_cache = load_learning_sessions()
    return {
        "total_patients": len(learning_cache),
        "patient_ids": list(learning_cache.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)