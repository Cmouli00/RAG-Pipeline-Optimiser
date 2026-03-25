import litellm
import json

async def evaluate_results(question, pipeline_outputs):
    """
    pipeline_outputs: List of {'name': 'Small_Chunks', 'answer': '...', 'context': '...'}
    """
    prompt = f"""
    You are a RAG Quality Auditor. 
    Question: {question}
    
    Compare these different RAG pipeline outputs:
    {pipeline_outputs}
    
    Score each out of 10 for:
    1. Faithfulness (Is it in the context?)
    2. Answer Relevance (Did it answer the user?)
    
    Return a JSON response:
    {{
        "winner": "Pipeline Name",
        "analysis": "Briefly why",
        "scores": [{{ "name": "...", "score": 8.5 }}]
    }}
    """
    
    response = await litellm.acompletion(
        model="ollama/phi3",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )

    response_json = response.choices[0].message.content
    return json.loads(response_json)
    