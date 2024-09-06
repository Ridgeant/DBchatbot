from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from pydantic import BaseModel
import sqlite3
import pandas as pd
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Any
import json
import os
import tempfile

app = FastAPI()

# Global variables to store database connection and LangChain components
db_conn = None
db_uri = None
llm = None
api_key = None

class SQLQuery(BaseModel):
    query: str

class QueryResult(BaseModel):
    query: str
    result: List[Dict[str, Any]]

class GraphRequest(BaseModel):
    queries_results: List[QueryResult]

@app.post("/api")
async def set_api_key(api_key_header: str = Header(...)):
    global api_key, llm
    api_key = api_key_header
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return {"message": "API key set successfully"}

@app.post("/db")
async def upload_db(file: UploadFile = File(...)):
    global db_conn, db_uri
    
    # Create a temporary directory to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
        file_path = temp_file.name  # Full path of the saved file
        # Save the uploaded file to the temporary file
        temp_file.write(await file.read())
    
    # Connect to the SQLite database using the file path
    db_conn = sqlite3.connect(file_path, check_same_thread=False)
    db_uri = f"sqlite:///{file_path}"  # Use the file path for the URI
    
    return {"message": "Database uploaded and connected successfully", "file_path": file_path}

@app.post("/sql")
async def execute_sql(sql_query: SQLQuery):
    global db_conn, db_uri, llm
    
    if not db_conn:
        raise HTTPException(status_code=400, detail="Database not connected. Please upload a database first.")
    
    if not llm:
        if not api_key:
            raise HTTPException(status_code=400, detail="API key not set. Please set the API key first.")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    try:
        # Generate SQL query using LangChain
        db = SQLDatabase.from_uri(db_uri)
        context = db.get_context()
        chain = create_sql_query_chain(llm, db)

        # Define the template string
        template_string = '''{dynamic_template}

        Only use the following tables and schema:
        {table_info}

        Question: {input}

        Note: Only return the SQL query, no need for explanation or output.
        '''

        # Extract the dynamic template from the chain
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = '\n'.join(ans[:5])

        # Create the prompt using the template
        prompt = PromptTemplate(
            template=template_string,
            input_variables=['input', 'table_info'],
            partial_variables={'dynamic_template': dynamic_template}
        )

        # Format the prompt with the user query and table info
        _input = prompt.format_prompt(input=sql_query.query, table_info=context["table_info"])
        generated_query = llm.call_as_llm(_input.to_string())

        # Split the generated query into separate queries if necessary
        queries = [q.strip() for q in generated_query.split(';') if q.strip()]
        
        results = []
        for query in queries:
            try:
                df = pd.read_sql_query(query, db_conn)
                result = df.to_dict(orient="records")  # Convert to list of dicts
                results.append({"query": query, "result": result})
            except Exception as e:
                results.append({"query": query, "error": str(e)})
        
        return {"queries_results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph")
async def generate_graph(request: GraphRequest):
    global llm
    
    if not llm:
        if not api_key:
            raise HTTPException(status_code=400, detail="API key not set. Please set the API key first.")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    if not request.queries_results:
        raise HTTPException(status_code=400, detail="No query results found.")

    plotly_codes = []
    for query_result in request.queries_results:
        result = query_result.result
        df = pd.DataFrame(result)

        try:
            plotly_code = graph(df)
            plotly_codes.append(plotly_code)
        except Exception as e:
            plotly_codes.append(f"Error generating plot: {str(e)}")

    return {"plotly_codes": plotly_codes}


def graph(df):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Template string for Plotly code
    template_string = '''Generate a Plotly chart code suitable for visualizing this dataframe: {input}

    Note: Only return the code for the Plotly chart, and use 'df' as the DataFrame reference in the code.
    '''

    prompt = PromptTemplate(
        template=template_string,
        input_variables=['input'],
    )

    _input = prompt.format_prompt(input=df.to_dict())
    output = llm.call_as_llm(_input.to_string())

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
