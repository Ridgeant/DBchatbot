import streamlit as st
import sqlite3
import pyodbc
import psycopg2
import mysql.connector
import mariadb
import cx_Oracle
import duckdb
import google.auth
from google.cloud import bigquery
import clickhouse_driver
import prestodb
import crate.client
import pandas as pd
import plotly.graph_objs as go
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
import os

# Initialize session state variables
if 'db' not in st.session_state:
    st.session_state.db = None
if 'conn' not in st.session_state:
    st.session_state.conn = None
if 'db_type' not in st.session_state:
    st.session_state.db_type = None

def main():
    st.title("SQL Query Executor with LangChain")

    # Set up the environment variable for the OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Select the database type
    db_type = st.selectbox("Select Database Type", ["SQLite", "SQL Server (MSSQL)", "PostgreSQL", "MySQL", "MariaDB", 
                                                    "Oracle", "DuckDB", "Google BigQuery", "ClickHouse", "PrestoDB", 
                                                    "CrateDB"])

    # File uploader for the SQLite DB file (only visible if SQLite is selected)
    db_file = None
    if db_type == "SQLite":
        db_file = st.file_uploader("Upload your SQLite database file", type=["db", "sqlite"])

    # SQL Server (MSSQL) connection details
    if db_type == "SQL Server (MSSQL)":
        sql_server_details = {
            "server": st.text_input("SQL Server Name"),
            "database": st.text_input("Database Name"),
            "username": st.text_input("Username"),
            "password": st.text_input("Password", type="password")
        }

    # PostgreSQL connection details
    if db_type == "PostgreSQL":
        postgres_details = {
            "host": st.text_input("PostgreSQL Host"),
            "database": st.text_input("Database Name"),
            "username": st.text_input("Username"),
            "password": st.text_input("Password", type="password"),
            "port": st.text_input("Port", value="5432")
        }

    # MySQL connection details
    if db_type == "MySQL":
        mysql_details = {
            "host": st.text_input("MySQL Host"),
            "database": st.text_input("Database Name"),
            "username": st.text_input("Username"),
            "password": st.text_input("Password", type="password"),
            "port": st.text_input("Port", value="3306")
        }

    # MariaDB connection details
    if db_type == "MariaDB":
        mariadb_details = {
            "host": st.text_input("MariaDB Host"),
            "database": st.text_input("Database Name"),
            "username": st.text_input("Username"),
            "password": st.text_input("Password", type="password"),
            "port": st.text_input("Port", value="3306")
        }

    # Oracle connection details
    if db_type == "Oracle":
        oracle_details = {
            "dsn": st.text_input("Oracle DSN (Data Source Name)"),
            "username": st.text_input("Username"),
            "password": st.text_input("Password", type="password")
        }

    # DuckDB connection details
    if db_type == "DuckDB":
        duckdb_file = st.file_uploader("Upload your DuckDB database file", type=["duckdb"])

    # Google BigQuery connection details
    if db_type == "Google BigQuery":
        gcp_credentials_file = st.file_uploader("Upload your GCP credentials JSON file", type=["json"])
        gcp_project_id = st.text_input("GCP Project ID")
        if gcp_credentials_file:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_credentials_file.name

    # ClickHouse connection details
    if db_type == "ClickHouse":
        clickhouse_details = {
            "host": st.text_input("ClickHouse Host"),
            "database": st.text_input("Database Name"),
            "username": st.text_input("Username"),
            "password": st.text_input("Password", type="password"),
            "port": st.text_input("Port", value="9000")
        }

    # PrestoDB connection details
    if db_type == "PrestoDB":
        prestodb_details = {
            "host": st.text_input("PrestoDB Host"),
            "catalog": st.text_input("Catalog"),
            "schema": st.text_input("Schema"),
            "username": st.text_input("Username"),
            "port": st.text_input("Port", value="8080")
        }

    # CrateDB connection details
    if db_type == "CrateDB":
        cratedb_details = {
            "host": st.text_input("CrateDB Host"),
            "port": st.text_input("Port", value="4200")
        }

    if st.button("Connect"):
        st.session_state.db_type = db_type
        if db_type == "SQLite" and db_file:
            st.success("SQLite database file uploaded successfully!")
            st.session_state.conn = sqlite3.connect(db_file.name, check_same_thread=False)
            st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_file.name}")
        elif db_type == "SQL Server (MSSQL)" and sql_server_details:
            conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={sql_server_details["server"]};DATABASE={sql_server_details["database"]};UID={sql_server_details["username"]};PWD={sql_server_details["password"]}'
            st.session_state.conn = pyodbc.connect(conn_str)
            st.success(f"Connected to SQL Server database: {sql_server_details['database']}")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "PostgreSQL" and postgres_details:
            st.session_state.conn = psycopg2.connect(
                host=postgres_details["host"],
                database=postgres_details["database"],
                user=postgres_details["username"],
                password=postgres_details["password"],
                port=postgres_details["port"]
            )
            st.success(f"Connected to PostgreSQL database: {postgres_details['database']}")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "MySQL" and mysql_details:
            st.session_state.conn = mysql.connector.connect(
                host=mysql_details["host"],
                database=mysql_details["database"],
                user=mysql_details["username"],
                password=mysql_details["password"],
                port=mysql_details["port"]
            )
            st.success(f"Connected to MySQL database: {mysql_details['database']}")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "MariaDB" and mariadb_details:
            st.session_state.conn = mariadb.connect(
                host=mariadb_details["host"],
                database=mariadb_details["database"],
                user=mariadb_details["username"],
                password=mariadb_details["password"],
                port=mariadb_details["port"]
            )
            st.success(f"Connected to MariaDB database: {mariadb_details['database']}")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "Oracle" and oracle_details:
            st.session_state.conn = cx_Oracle.connect(
                user=oracle_details["username"],
                password=oracle_details["password"],
                dsn=oracle_details["dsn"]
            )
            st.success(f"Connected to Oracle database: {oracle_details['dsn']}")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "DuckDB" and duckdb_file:
            st.session_state.conn = duckdb.connect(duckdb_file.name)
            st.success("Connected to DuckDB database.")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "Google BigQuery" and gcp_project_id:
            credentials, project = google.auth.default()
            client = bigquery.Client(project=gcp_project_id)
            st.success("Connected to Google BigQuery.")
            st.session_state.db = SQLDatabase.from_client(client)
        elif db_type == "ClickHouse" and clickhouse_details:
            st.session_state.conn = clickhouse_driver.Client(
                host=clickhouse_details["host"],
                user=clickhouse_details["username"],
                password=clickhouse_details["password"],
                database=clickhouse_details["database"],
                port=int(clickhouse_details["port"])
            )
            st.success(f"Connected to ClickHouse database: {clickhouse_details['database']}")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "PrestoDB" and prestodb_details:
            st.session_state.conn = prestodb.dbapi.connect(
                host=prestodb_details["host"],
                port=int(prestodb_details["port"]),
                user=prestodb_details["username"],
                catalog=prestodb_details["catalog"],
                schema=prestodb_details["schema"]
            )
            st.success(f"Connected to PrestoDB catalog: {prestodb_details['catalog']}, schema: {prestodb_details['schema']}")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        elif db_type == "CrateDB" and cratedb_details:
            st.session_state.conn = crate.client.connect(f"http://{cratedb_details['host']}:{cratedb_details['port']}")
            st.success("Connected to CrateDB database.")
            st.session_state.db = SQLDatabase.from_connection(st.session_state.conn)
        else:
            st.warning("Please provide all necessary connection details.")
            return

    if st.session_state.db:
        context = st.session_state.db.get_context()

        ques = questions(st.session_state.db, context)
        st.write("Sample Questions:")
        st.code(ques)

        # Enter SQL query
        user_input = st.text_area("Enter your SQL query")

        if st.button("Run Query"):
            output = run_query(user_input, st.session_state.db, context)
            st.write("Generated SQL Queries:")
            st.code(output)

            # Execute and display the query results
            queries = output.split("\n\n")  # Assuming the queries are separated by double newlines
            for i, query in enumerate(queries, start=1):
                st.write(f"### Query {i} Results:")
                df = display_results(query, st.session_state.conn)
                if df is not None:
                    st.dataframe(df)

                    # Generate the graph for each query result
                    fig = graph(df)  # Now returns the Plotly figure directly
                    if fig:
                        st.plotly_chart(fig)

def run_query(user_input, db, context):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature="0")

    # Dynamic prompt
    chain = create_sql_query_chain(llm, db)
    k = chain.get_prompts()[0]
    ans = k.template.split("\n")
    dynamic_template = ans[:5]
    dynamic_template = '\n'.join(dynamic_template)

    # Template string
    template_string = ''' {dynamic_template}\

    Only use the following tables and schema\
    {table_info}\

    question : {input}

    note : only give SQL query no need to give explanation and output
    '''

    prompt = PromptTemplate(
            template=template_string,
            input_variables=['input', 'table_info'],
            partial_variables={'dynamic_template': dynamic_template}
        )

    _input = prompt.format_prompt(input=user_input, table_info=context["table_info"])

    output = llm.call_as_llm(_input.to_string())
    return output

def display_results(output, conn):
    c = conn.cursor()

    try:
        for query in output.split("\n\n"):
            c.execute(query)

            columns = [description[0] for description in c.description]
            results = c.fetchall()

            # Create DataFrame and display
            df = pd.DataFrame(results, columns=columns)
            return df
    except (sqlite3.Error, pyodbc.Error, psycopg2.Error, mysql.connector.Error, mariadb.Error, cx_Oracle.Error, 
            clickhouse_driver.errors.Error, prestodb.exceptions.DatabaseError, crate.client.exceptions.ConnectionError) as e:
        st.error(f"An error occurred: {e}")
        return None
    finally:
        c.close()

def questions(db, context):
  from langchain.chains import create_sql_query_chain
  from langchain_openai import ChatOpenAI
  from langchain.prompts import PromptTemplate

  llm = ChatOpenAI(model="gpt-3.5-turbo", temperature="0")

  # dynamic prompt
  chain = create_sql_query_chain(llm, db)
  k = chain.get_prompts()[0]
  ans = k.template.split("\n")
  dynamic_template = ans[:5]
  dynamic_template = '\n'.join(dynamic_template)
  # end dynamic prompt

  template_string = ''' {dynamic_template}\

  Only use the following tables and schema\
  {table_info}\

  so your duty is generate the 25 sample question based on the {table_info} for sql queries and dashboars KPIS\

  note : only give list of question in list formate no need to give explaination.
  '''
  
  prompt = PromptTemplate(
          template=template_string,
          input_variables=['table_info'],
          partial_variables = {'dynamic_template': dynamic_template}
      )

  _input = prompt.format_prompt(table_info = context["table_info"])

  output = llm.call_as_llm(_input.to_string())
  return output

def graph(df):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature="0")

    # Dynamic prompt
    chain = create_sql_query_chain(llm, st.session_state.db)
    k = chain.get_prompts()[0]
    ans = k.template.split("\n")
    dynamic_template = ans[:5]
    dynamic_template = '\n'.join(dynamic_template)

    # Template string
    template_string = '''Generate a dictionary that can be used to create a Plotly chart for visualizing this dataframe. The dictionary should be suitable to be passed directly to `go.Figure()`:

    DataFrame: {input}

    Ensure the returned object is a dictionary that can be used with `go.Figure()`.
    
    directly print the code from import no need to put ```python and extra things.
    '''

    prompt = PromptTemplate(
        template=template_string,
        input_variables=['input'],
    )

    _input = prompt.format_prompt(input=df)

    # Get the generated dictionary from LLM
    plotly_dict = llm.call_as_llm(_input.to_string())

    # Convert the generated string to a Python dictionary
    plotly_dict = eval(plotly_dict)

    # Return the Plotly figure object created from the dictionary
    fig = go.Figure(plotly_dict)
    return fig

if __name__ == "__main__":
    main()
