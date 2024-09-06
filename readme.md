command for fast api:
python main.py

Endpoints: db, sql, graph, api

db:
Body: form-data then Key:file, Type: File, Value: name.db

api:
Headers: key: api-key-header, value: your_api_key

sql:
Body: raw and select JSON
example: {
    "query":"what is the revenue per project in 2022 ? "
}

graph:
Body: raw and select JSON and put the output of /sql into this

-----------------------------------------------------------
command for Streamlit:
streamlit run streamlit.py 
