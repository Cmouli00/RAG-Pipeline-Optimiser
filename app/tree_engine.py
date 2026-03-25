import os
import asyncio
import litellm

async def summarize_batch_async(batch: list, batch_index: int) -> str:
    """Asynchronous call to Phi-3 for a single batch of text."""
    context_text = "\n\n".join(batch)
    prompt = f"Summarize these technical excerpts into one high-level theme:\n\n{context_text}"
    
    try:
        # We use await here so the program can move to the next batch while waiting
        response = await litellm.acompletion(
            model="ollama/phi3",
            messages=[{"role": "user", "content": prompt}],
            timeout=60
        )
        summary = response.choices[0].message.content.strip()
        return f"[SECTION {batch_index} SUMMARY]: {summary}"
    except Exception as e:
        return f"[SECTION {batch_index} ERROR]: {str(e)}"

async def build_tree_context_async(docs):
    """
    Groups chunks and summarizes them in parallel to build the RAPTOR tree.
    """
    leaf_texts = [d.page_content for d in docs]
    batch_size = 5
    batches = [leaf_texts[i:i + batch_size] for i in range(0, len(leaf_texts), batch_size)]
    
    # Trigger all summarizations at the same time
    tasks = [summarize_batch_async(batch, i+1) for i, batch in enumerate(batches)]
    summaries = await asyncio.gather(*tasks)
    
    # Return original granular details + high-level summaries
    return leaf_texts + summaries