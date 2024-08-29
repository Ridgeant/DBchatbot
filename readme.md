command for fast api:
python main.py

Endpoints: db, sql, graph

db:
Body: Key:file, Type: File, Value: name.db

sql:
Body: raw and select JSON
example: {
    "query":"what is the revenue per project in 2022 ? "
}

graph:
Body: raw and select JSON
exmple:{
    "data": [
        {
            "ProjectId": 1,
            "VendorName": "Raju"
        },
        {
            "ProjectId": 2,
            "VendorName": "Rajesh",
        }
    ]
}

-----------------------------------------------------------
command for Streamlit:
streamlit run streamlit.py 
