from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import sqlite3
import pandas as pd
import plotly.graph_objs as go
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Any
import json

app = FastAPI()

# Global variables to store database connection and LangChain components
db_conn = None
db_uri = None
llm = None

class SQLQuery(BaseModel):
    query: str

class GraphRequest(BaseModel):
    # x_axis: str
    # y_axis: str
    # graph_type: str
    data: List[Dict[str, Any]]  # Use Any to allow mixed types

@app.post("/db")
async def upload_db(file: UploadFile = File(...)):
    global db_conn, db_uri
    
    # Save the uploaded file
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    
    # Connect to the database
    db_conn = sqlite3.connect(file.filename, check_same_thread=False)
    db_uri = f"sqlite:///{file.filename}"
    
    return {"message": "Database uploaded and connected successfully"}

@app.post("/sql")
async def execute_sql(sql_query: SQLQuery):
    global db_conn, db_uri, llm
    
    if not db_conn:
        raise HTTPException(status_code=400, detail="Database not connected. Please upload a database first.")
    
    if not llm:
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

        # Execute the generated SQL query
        df = pd.read_sql_query(generated_query, db_conn)
        
        # Convert DataFrame to JSON
        result = df.to_json(orient="records")
        
        return {"query": generated_query, "result": json.loads(result)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph")
async def generate_graph(request: GraphRequest):
    global llm
    
    if not llm:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # x_axis = request.x_axis
    # y_axis = request.y_axis
    # graph_type = request.graph_type
    data = request.data

    df = pd.DataFrame(data)

    try:
        plotly_code = graph(df)
        return {"plotly_code": plotly_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
