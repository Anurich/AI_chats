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
Make sure you provide concise answer. representation of answer should be in human readable form.
Context: {context}
Question: {question}
Answer:


"""



map_template="""Please summarize the following content concisely. Use only the information provided, without adding any new content. The summary should be in bullet points, limited to 5 points. Each point should clearly reflect the main ideas from the provided content.
Format the summary as follows:
1.
2.
3.
4.
5.
Content:
{docs}
"""
# TOKEN_SENTIMENT_PROMPT = """
# Your task involves two parts: token classification and sentiment analysis.
# As an expert in token classification and semantic understanding, your task is to thoroughly analyze the provided content and identify the most relevant tokens that encapsulate its main points. List the 5 most pertinent tokens in bullet points. If no relevant tokens are found, please indicate accordingly
# Sentiment Analysis:
# As an annotator, your job is to assess the sentiment of the given content. Dive deep into its meaning and determine whether the sentiment is positive or negative. Provide a brief explanation (not exceeding 20 tokens) for your sentiment assignment.
# Content: {content}
# Always address both parts of the task: token classification and sentiment analysis.
# """


TOKEN_SENTIMENT_PROMPT = """
Content Insights
Analyze the given content to uncover key insights.

Key Tokens:
Extract the top 5 tokens (words or phrases) that convey the main ideas and concepts. Present them in a bullet-point list.

Sentiment Snapshot:
Assess the overall sentiment of the content as:
Positive (e.g., enthusiastic, optimistic)
Negative (e.g., critical, pessimistic)
Neutral (e.g., informative, objective)
Provide a brief explanation (max 20 tokens) for your sentiment assignment.

Content: {content}

Output Format:
Key Tokens:
Token 1
Token 2
Token 3
Token 4
Token 5
Sentiment: [Positive/Negative/Neutral]
Explanation: [brief explanation]
Tips:
1. Focus on tokens that carry significant meaning and context.
2. Consider the tone, language, and intent behind the content.
3. Keep your explanation concise and clear.

By following this format, you'll provide valuable insights into the content's key tokens and sentiment.
"""