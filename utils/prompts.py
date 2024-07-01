from langchain import hub



MAIN_PROMPT =  hub.pull("hwchase17/openai-functions-agent")
RAG_FUSION  =  """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (2 queries):"""

CHAT_WITH_TABLE = """
Your expertise lies in deciphering complex table data, even when it's presented in a less-than-ideal format. Your mission is to extract meaningful insights from the provided data, organized by table numbers.
Each table is delineated by its number, followed by sequential data entries. For example, Table 1 encompasses information from the subsequent lines until the next table is encountered.
In cases where a table lacks information, respond with "no information found for that table." Your analysis should aim to uncover valuable insights tailored to each table, steering clear of irrelevant or generic responses.
Always present the extracted data in a clear and organized manner using markdown table format to facilitate understanding.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:

Example:

Table 1:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

Table 2:
| Column A | Column B | Column C |
|----------|----------|----------|
| Info A   | Info B   | Info C   |
| Info D   | Info E   | Info F   |

and so on.
"""


ROUTER = """
    You are a specialist in assigning topics to the query provided to you. You will assign only one of the following two topics:
    1. chat_with_table 
    2. chat_with_pdf 

    Provided the query: {query}, you will interpret based on the context if this query is related to chat_with_pdf or chat_with_table.
    
    Criteria:
    - Assign the topic 'chat_with_table' if the query contains the keyword 'table' or refers to data that is structured in rows and columns.
    - Assign the topic 'chat_with_pdf' for all other queries.

    Do not assign any other topics except the two discussed above. 
    he answer should be given in the following JSON format:
        "topic": "chat_with_table" 
    or
        "topic": "chat_with_pdf"
"""

CHAT_WITH_PDF="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Provide the answer in great detail. After the answer, list the important key points, numbered 1 through 5.
Context: {context}
Question: {question}
Answer:

Key Points:
1. [First key point]
2. [Second key point]
3. [Third key point]
4. [Fourth key point]
5. [Fifth key point]
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
TOKEN_SENTIMENT_PROMPT = """
Your task involves two parts: token classification and sentiment analysis.
As an expert in token classification and semantic understanding, your task is to thoroughly analyze the provided content and identify the most relevant tokens that encapsulate its main points. List the 5 most pertinent tokens in bullet points. If no relevant tokens are found, please indicate accordingly
Sentiment Analysis:
As an annotator, your job is to assess the sentiment of the given content. Dive deep into its meaning and determine whether the sentiment is positive or negative. Provide a brief explanation (not exceeding 20 tokens) for your sentiment assignment.
Content: {content}
Always address both parts of the task: token classification and sentiment analysis.
"""


