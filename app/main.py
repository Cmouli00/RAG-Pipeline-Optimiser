from fastapi import FastAPI, UploadFile, File
import asyncio

from fastapi.encoders import jsonable_encoder
from .engine import run_rag_strategy
from .evaluator import evaluate_results

app = FastAPI()

@app.post("/optimize")
async def optimize_rag(question: str, file: UploadFile = File(...)):
    try:
        # 1. Save file locally
        temp_path = f"temp_{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 2. Run Pipelines in Parallel
        strategies = ["Small_Chunks", "Large_Chunks", "Hybrid_Mock", "Advanced_Rerank", "Tree_RAPTOR"]
        
        # FIX: Call the async function directly. 
        # Do NOT use asyncio.to_thread if run_rag_strategy is 'async def'
        tasks = [run_rag_strategy(temp_path, question, s) for s in strategies]
        
        # Use return_exceptions=True so one failure (like RAPTOR) doesn't kill the whole request
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Compile for Judge
        outputs = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                # This prevents the "coroutine is not iterable" error in the encoder
                print(f"Error in {strategies[i]}: {res}")
                outputs.append({"name": strategies[i], "context": f"Error: {str(res)}"})
            else:
                outputs.append({"name": strategies[i], "context": res})

        # 4. Get the Winner
        evaluation = await evaluate_results(question, outputs)

        return {"results": outputs, "evaluation": evaluation}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": "Internal Server Error", "details": str(e)}