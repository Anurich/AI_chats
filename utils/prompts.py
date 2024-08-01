from langchain import hub



MAIN_PROMPT =  hub.pull("hwchase17/openai-functions-agent")
RAG_FUSION  =  """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (2 queries):"""


CATEGORIZATION = """
Given the context of the document, assign the appropriate category and color to each context provided below. The categories to choose from are:

Finance/Banking red
Resume blue
Education green
Environment orange
Health pink
Entertainment brown
Legal/Documents purple
Others black

Ensure you select only from the categories listed above. If the category is unclear, assign it to point number 6 (Others). Your answer should be one of the points above.
Do not provide any kind of explaination, I just need category and it's color nothing else, 
for example 
Finance/Banking red
Finance/Banking red
Resume blue
Education green
Environment orange
Health pink
Entertainment brown
Legal/Documents purple
Others black

Context: {Context}
Answer:
"""

KEY_POINTS = """
Please extract the list of key points from the context provided. Focus on important topics and ensure each key point is no more than 2 tokens. Do not include numeric values.
## Example
Context: My name is Anupam, I have a loan amount of 200, I live in Noida, and I have a car.

Key Points:
name
loan amount
live
car

Focus on the entity that represents the question, not the answer. Skip any numeric values, floating points, or doubles.You will be provided with the context below:
Context:
{Context}
Key Points:
"""




CHAT_WITH_TABLE = """
Your expertise lies in deciphering complex table data, even when it's presented in a less-than-ideal format. Your mission is to extract meaningful insights from the provided data, organized by table numbers.
Each table is delineated by its number, followed by sequential data entries. For example, filename-Table 1 encompasses information from the subsequent lines until the next table is encountered.
In cases where a table lacks information, respond with "no information found for that table." Your analysis should aim to uncover valuable insights tailored to each table, steering clear of irrelevant or generic responses.
Always present the extracted data in a clear and organized manner using HTML table format to facilitate understanding.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:

Example:
    <span> <b>filename-Table 1</b> </span>
    <table border="1">
    <tr>
        <th>Column 1</th>
        <th>Column 2</th>
        <th>Column 3</th>
    </tr>
    <tr>
        <td>Data 1</td>
        <td>Data 2</td>
        <td>Data 3</td>
    </tr>
    <tr>
        <td>Data 4</td>
        <td>Data 5</td>
        <td>Data 6</td>
    </tr>
    </table>

    <br/>
    <br/>
    <span> <b>filename-Table 2</b> </span>
    <table border="1">
    <tr>
        <th>Column A</th>
        <th>Column B</th>
        <th>Column C</th>
    </tr>
    <tr>
        <td>Info A</td>
        <td>Info B</td>
        <td>Info C</td>
    </tr>
    <tr>
        <td>Info D</td>
        <td>Info E</td>
        <td>Info F</td>
    </tr>
    </table>

and so on.
Make sure the response of the table is formatted in HTML table tags with proper functioning code.
"""



ROUTER = """
    You are an expert in determining the appropriate topic for a given query. You have two topics to choose from:

    chat_with_table
    chat_with_pdf
    Your task is to assign one of these topics based on the query provided and the number of tables available.

    Criteria for assignment:

    If the query contains the keyword 'table' and the number of tables available is greater than 0, assign the topic 'chat_with_table'.
    For all other queries, including those without the keyword 'table' or where the number of tables available is less than or equal to 0, assign the topic 'chat_with_pdf'.
    You must choose one of the two topics and provide the answer in the following JSON format:
            "topic": "chat_with_table" 
        or
            "topic": "chat_with_pdf"
    
    Here is the query for your interpretation: {query}
    Number of tables available: {table}
"""

CHAT_WITH_PDF="""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Make sure to provide detailed responses to any question. The answer should be in a human-readable form.
###
for example 
if response  in the answer is in points it should be like:
    1. response1 \n
    2. response2 \n
    3. response3 \n
if response in the answer is in paragraphs it should be like:
    Paragraph1 \n
    \n
    Paragraph2 \n
###
Apart from this, you should also return the sentiment of the answer generated by you. Make sure to provide the sentiment and a max 20-token explanation.
Based on the language of the context and question, the response should be in the same language.

Context: {context}
Question: {question}
Answer:
Sentiment:
Explanation:

"""




map_template="""Please summarize the following content concisely. Use only the information provided, without adding any new content. The summary should be in bullet points, limited to 5 points. Each point should clearly reflect the main ideas from the provided content.
Format the summary as follows:
1.
2.
3.
4.
5.
Note: only provide the available points. Do not include empty or placeholder points.
Content:
{docs}
"""

FILE_SEARCH_PROMPT = """
You are a relevance evaluator. Your task is to assess the connection between a given context and description.
Evaluate the relevance by checking if the description is related to the context, and provide the probability score.

Evaluate the relevance by checking:
- Entity match
- Topic alignment
- Key phrase overlap
- Semantic similarity

You will be provided with: 
pdf_name: {pdf_name}
Context: {Context}
Description: {description}


Your tasks are:
1. check the relevance between description and context.
2. Calculate the probability (0-1) of the description matching the context.
3. Extract the answer from the context based on  description, and also provide the 20 token explaination. 

** Important ** 
1. output should be in the format as shown below
 {{"pdf_name" : "probability" : "explaination"}}
2. Do NOT use any other format.
3. Ensure your explanation is concise and within the 20-token limit. Provide accurate and precise responses.

We will only accept responses in the specified format. Deviations will be considered incorrect.
"""
