from fastapi import FastAPI, UploadFile, File
import asyncio
from .engine import run_rag_strategy
from .evaluator import evaluate_results

app = FastAPI()

@app.post("/optimize")
async def optimize_rag(question: str, file: UploadFile = File(...)):
    # 1. Save file locally
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 2. Run Pipelines in Parallel
    strategies = ["Small_Chunks", "Large_Chunks", "Hybrid_Mock", "Advanced_Rerank"]
    
    # This is the "Magic": running all 4 experiments at once
    tasks = [asyncio.to_thread(run_rag_strategy, temp_path, question, s) for s in strategies]
    results = await asyncio.gather(*tasks)

    # 3. Compile for Judge
    outputs = []
    for i, res in enumerate(results):
        outputs.append({"name": strategies[i], "context": res})

    # 4. Get the Winner
    evaluation = evaluate_results(question, outputs)
    
    return {"results": outputs, "evaluation": evaluation}