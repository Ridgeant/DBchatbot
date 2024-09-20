import sqlite3
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
import os
import tempfile
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain import hub

class DatabaseQueryVisualizer:
    def __init__(self):
        self.db = None
        self.conn = None
        self.db_type = None
        self.query_result = None
        self.custom_graphs = {}
        self.sql_queries = None
        self.query_outputs = []
        self.context = None
        self.processed = False
        self.db_connected = False
        self.rag_chain_with_source = None
        

    def connect_to_database(self, db_type,openai_api_key, **connection_details):
        self.db_type = db_type
        
        if db_type == "SQLite":
            self.conn = sqlite3.connect(connection_details['db_file'], check_same_thread=False)
            self.db = SQLDatabase.from_uri(f"sqlite:///{connection_details['db_file']}")

        else:
            raise ValueError(f"Unsupported database type: {db_type}")


        if self.db:
            self.context = self.db.get_context()
            self.db_connected = True
            print(f"Connected to {db_type} database.")
        print("----------------------------------------------------------")
        sample_questions = self.questions(openai_api_key)
        print("Sample questions generated based on the database schema:")
        print(sample_questions)  # Display the generated sample questions
        print("----------------------------------------------------------")

    def questions(self,openai_api_key):
        from langchain.chains import create_sql_query_chain
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature="0",openai_api_key=openai_api_key)

        # Create the dynamic SQL query chain
        chain = create_sql_query_chain(llm, self.db)
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = ans[:5]
        dynamic_template = '\n'.join(dynamic_template)

        # Define the custom prompt for generating questions
        template_string = '''{dynamic_template}\

        Only use the following tables and schema\
        {table_info}\

        Your task is to generate 30 sample questions based on the {table_info} for SQL queries and dashboard KPIs.\

        Note: Only provide a list of questions in list format, no explanations needed.
        '''
        
        prompt = PromptTemplate(
            template=template_string,
            input_variables=['table_info'],
            partial_variables={'dynamic_template': dynamic_template}
        )

        # Use the context to format the prompt
        _input = prompt.format_prompt(table_info=self.context["table_info"])

        # Generate the output from the LLM
        output = llm.call_as_llm(_input.to_string())
        
        return output

    def process_pdf(self, pdf_files, openai_api_key):
        all_docs = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            data = loader.load()
            if not data:
                print(f"No text found in the PDF file: {pdf_file}")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(data)
                all_docs.extend(docs)

        if all_docs:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.from_documents(all_docs, embeddings)
            retriever = db.as_retriever()
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key=openai_api_key)
            prompt = hub.pull("rlm/rag-prompt")
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt
                | llm
                | StrOutputParser()
            )
            
            self.rag_chain_with_source = RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
            ).assign(answer=rag_chain_from_docs)

            self.processed = True
            print("PDF processing complete.")

    # def process_pdf_query(self, query):
    #     try:
    #         output = {}
    #         for chunk in self.rag_chain_with_source.stream(query):
    #             for key in chunk:
    #                 if key not in output:
    #                     output[key] = chunk[key]
    #                 else:
    #                     output[key] += chunk[key]
    #         return output['answer']
    #     except Exception as e:
    #         print(f"Error during PDF query processing: {e}")
    #         return None
    def process_pdf_query(self, query):
        try:
            output = {}
            retrieved_docs = []
            for chunk in self.rag_chain_with_source.stream(query):
                for key in chunk:
                    if key == "context":
                        retrieved_docs.append(chunk[key])  # Collect retrieved documents
                    if key not in output:
                        output[key] = chunk[key]
                    else:
                        output[key] += chunk[key]
            
            # Print the retrieved documents
              
            
            # Return the answer from the output
            return output.get('answer', ''), retrieved_docs
        except Exception as e:
            print(f"Error during PDF query processing: {e}")
            return None

    def run_query(self, user_input, openai_api_key):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature="0",openai_api_key=openai_api_key)
        chain = create_sql_query_chain(llm, self.db)
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = '\n'.join(ans[:5])

        template_string = '''
        {dynamic_template}

        Only use the following tables and schema
        {table_info}

        question : {input}

        note : only give SQL query no need to give explanation and output
        If you give more than one sql queries make sure it is seperated by two white spaces, no need to put ```python, ```sql etc. and extra things.
        If there is a Union or and other 2 query join condition like union then dont make a white space before or after on condition.
        If you give more than one sql queries make sure it is seperated by two white spaces and never include counting of sql queries like 1. 2. 3. each query is seprated by two white spaces
        If you give more than one sql queries make sure it is seperated by two white spaces and never include counting of sql queries like 1. 2. 3. each query is seprated by two white spaces
        '''

        prompt = PromptTemplate(
            template=template_string,
            input_variables=['input', 'table_info'],
            partial_variables={'dynamic_template': dynamic_template}
        )

        _input = prompt.format_prompt(input=user_input, table_info=self.context["table_info"])
        output = llm.call_as_llm(_input.to_string())
        return output

    def display_results(self, output):
        c = self.conn.cursor()
        try:
            for query in output.split("\n\n"):
                c.execute(query)
                columns = [description[0] for description in c.description]
                results = c.fetchall()
                df = pd.DataFrame(results, columns=columns)
                self.query_result = df
                self.query_outputs.append(df)
                print("----------------------------------------------------------")
                print("SQL Output")
                print("----------------------------------------------------------")
                print(df)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            c.close()

    def generate_graph(self, df, openai_api_key):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature="0",openai_api_key=openai_api_key)
        chain = create_sql_query_chain(llm, self.db)
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = '\n'.join(ans[:5])

        # template_string = '''
        # You are given the following DataFrame:
        # {input}

        # Generate a Plotly code to visualize this data. Ensure that the graph accurately represents the data and is informative. Only provide the code for the graph.

        # - If the DataFrame has more than one record, select the graph type based on the data columns (such as scatter, line, or bar charts). Avoid defaulting to a bar chart unless it's the most appropriate type.
        # - If the DataFrame has only one record with **multiple columns**, generate a KPI-style card using `plotly.graph_objects`. The card must display key **numeric values** from the DataFrame. If a column contains non-numeric data (e.g., text, dates), include it in the title or as a text annotation, but do not use it as the value for the KPI.
        # - If the DataFrame contains **non-numeric data** such as **dates or text**, do not pass them as the `value` in `go.Indicator`. Instead, display them in the title or as plain text annotations in the visualization.
        # - Ensure that only **numeric values** are passed to `go.Indicator` as the `value` property. For non-numeric data like strings or dates, use alternative methods such as titles, text annotations, or other relevant Plotly elements (e.g., `go.Text`).
        # - The DataFrame is already imported as 'df', so do not include import statements or redundant definitions.
        # - If a column contains non-numeric data (e.g., text, dates), include it in the title or as a text annotation, but **do not use it as the value** for the KPI.
        # - Ensure that only **numeric values** are passed to `go.Indicator` as the `value` property. For non-numeric data, such as strings or dates, use them in the title or text annotations, but **never in the value**.  
        # Return only the Plotly code for the graph, and No need to put ```python and extra things. also define all lib with code.
        # - Always include error handling to prevent crashes use try catch and in catch display a **card** written Graph not displayed. Please use Custom Graph.
        # '''
        template_string = """
        You are given the following DataFrame:
        {input}

        Generate a Plotly code to visualize this data. Ensure that the graph accurately represents the data and is informative. Only provide the code for the graph.

        - Analyze the DataFrame columns and choose the **most appropriate graph type**:
            - For numeric columns with a clear relationship, use scatter plots.
            - For categorical data with counts or relationships, use bar charts.
            - For date or time-related data, use line charts or time series visualizations.
        - If the DataFrame contains **multiple numeric columns**, try to use charts like scatter, line, or other suitable types that show relationships between them.
        - Avoid defaulting to bar charts unless the data is categorical and best suited for it.
        - If the DataFrame has only one record with **multiple columns**, generate a KPI-style card using `plotly.graph_objects`. The card must display key **numeric values** from the DataFrame.
            - Handle numeric data correctly.
            - If a column contains non-numeric data (e.g., text, dates), include it in the title or as a text annotation, but **do not use it as the value** for the KPI.
        - Ensure that only **numeric values** are passed to `go.Indicator` as the `value` property. For non-numeric data like strings or dates, use alternative methods such as titles, text annotations, or other relevant Plotly elements (e.g., `go.Text`)
        - - Always include error handling to prevent crashes use try catch and in catch display a **card** written Graph not displayed. Please use Custom Graph.
        """

        prompt = PromptTemplate(
            template=template_string,
            input_variables=['input'],
        )

        _input = prompt.format_prompt(input=df)
        output = llm.call_as_llm(_input.to_string())
        return output

    def process_query(self, query, openai_api_key):
        if self.db_connected and not self.processed:
            # Database query
            self.query_outputs.clear()
            self.sql_queries = None
            self.custom_graphs.clear()
            output = self.run_query(query, openai_api_key)
            print("----------------------------------------------------------")
            print("SQL Query\n")
            print(output)
            print("----------------------------------------------------------")
            self.sql_queries = output
            self.display_results(output)
            for df in self.query_outputs:
                graph_code = self.generate_graph(df,openai_api_key)
                if "```python" in graph_code:
                    graph_code = graph_code.replace("```python", "").replace("```", "")
                print("----------------------------------------------------------")
                print("Generated Graph Code:")
                print(graph_code)
                print("----------------------------------------------------------")
                local_namespace = {"df": df, "go": go}
                exec(graph_code, local_namespace)
                print("----------------------------------------------------------")
                print("Custom Graph")
                x_axis = input(f'Select X-axis from {df.columns}')
                y_axis = input(f'Select Y-axis from {df.columns}')
                graph_type = input("Select from this [\"Line\", \"Bar\", \"Scatter\", \"Pie\"]")
                # generate_custom_graph(df, graph_type, x_axis, y_axis)
                custom_graph = self.generate_custom_graph(df, graph_type, x_axis, y_axis)
                custom_graph.show()  

            # Extract the figure object and display it in Streamlit
            # fig = local_namespace.get("fig")
            # if fig:
            #     print("Image is there")
                # Execute graph code and display graph (implementation depends on your environment)
        elif self.processed and not self.db_connected:
            # PDF query
            pdf_answer, retrieved_docs = self.process_pdf_query(query)
            print("----------------------------------------------------------")
            print("Retrieved Documents:")
            print("----------------------------------------------------------")
            for doc in retrieved_docs:
                print(doc)
            print("----------------------------------------------------------")
            print("PDF Answer:")
            print("----------------------------------------------------------")
            print(pdf_answer)
            print("----------------------------------------------------------")
            
        elif self.processed and self.db_connected:
            # Both PDF and database query
            pdf_answer, retrieved_docs = self.process_pdf_query(query)
            db_answer = self.run_query(query, openai_api_key)
            best_answer, from_db = self.compare_answers(query, pdf_answer, db_answer, openai_api_key)
            # print(best_answer)
            if from_db:
                print("----------------------------------------------------------")
                print("SQL Query\n")
                print(best_answer)
                print("----------------------------------------------------------")
                self.sql_queries = best_answer
                self.display_results(best_answer)
                for df in self.query_outputs:
                    graph_code = self.generate_graph(df, openai_api_key)
                    if "```python" in graph_code:
                        graph_code = graph_code.replace("```python", "").replace("```", "")
                    print("----------------------------------------------------------")
                    print("Generated Graph Code:")
                    print(graph_code)
                    print("----------------------------------------------------------")
                    local_namespace = {"df": df, "go": go}
                    exec(graph_code, local_namespace)
                    print("----------------------------------------------------------")
                    print("Custom Graph")
                    x_axis = input(f'Select X-axis from {df.columns}')
                    y_axis = input(f'Select Y-axis from {df.columns}')
                    graph_type = input("Select from this [\"Line\", \"Bar\", \"Scatter\", \"Pie\"]")
                    # generate_custom_graph(df, graph_type, x_axis, y_axis)
                    custom_graph = self.generate_custom_graph(df, graph_type, x_axis, y_axis)
                    custom_graph.show()  

            # Extract the figure object and display it in Streamlit
                # fig = local_namespace.get("fig")
                # if fig:
                #     print("Image is there")
            else:
                print("----------------------------------------------------------")
                print("Retrieved Documents:")
                print("----------------------------------------------------------")
                for doc in retrieved_docs:
                    print(doc)
                print("----------------------------------------------------------")
                print("PDF Answer:")
                print("----------------------------------------------------------")
                print(best_answer)
                print("----------------------------------------------------------")
                
    
    def generate_custom_graph(self, df, graph_type, x_axis, y_axis):
        if graph_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif graph_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif graph_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif graph_type == "Pie":
            fig = px.pie(df, names=x_axis, values=y_axis, title=f"{y_axis} distribution")
        return fig

    def compare_answers(self, query, pdf_answer, db_answer, openai_api_key):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key=openai_api_key)
        prompt_template = """
        You are provided with two answers to the following query:
        Query: {query}

        Answer from PDF:
        {pdf_answer}

        Answer from Database:
        {db_answer}

        Please determine which of the two answers is more suitable.
        Just write the exact same answer.
        Check perfectly for the answer from which it suits more.
        If the answer is from the Database add '[DB]' at the end
        and if it is from pdf dont't add anything at the end
        """
        prompt = PromptTemplate(
            input_variables=["query", "pdf_answer", "db_answer"],
            template=prompt_template
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({
            "query": query,
            "pdf_answer": pdf_answer,
            "db_answer": db_answer
        })
        best_answer = response.strip()
        from_db = best_answer.endswith("[DB]")
        if from_db:
            best_answer = best_answer.replace("[DB]", "").strip()
        return best_answer, from_db
    
# Example usage
if __name__ == "__main__":
    openai_api_key = input("Enter your api_key: ")
    
    visualizer = DatabaseQueryVisualizer()
    visualizer.connect_to_database("SQLite", db_file="BIRD.db", openai_api_key=openai_api_key)
    
    visualizer.process_pdf(["yolov7.pdf"], openai_api_key=openai_api_key)

    visualizer.process_query("What is the total budgeted cost for all tasks? ",openai_api_key=openai_api_key)