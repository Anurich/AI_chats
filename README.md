# AI Chat

## Overview
AI Chat is an advanced document interaction tool designed to enhance productivity and efficiency in handling document-based tasks. It enables users to upload, categorize, summarize, and extract key insights from documents while providing features like PDF comparison, keyword extraction, and knowledge graph building. The project leverages LangChain for foundational workflows and aims to address real-world limitations encountered during its development.

## Key Features
- *Upload Documents*: Easily upload documents for analysis and processing.
- *Categorize and Organize*: Automatically categorize and organize uploaded files.
- *Summarize Content*: Generate concise summaries for documents to quickly grasp their core ideas.
- *Extract Tables*: Export tables from documents into Excel files for better data handling.
- *Identify Top Keywords*: Extract and rank the most relevant keywords from documents.
- *Interactive Q&A*: Ask questions to single or multiple PDFs to retrieve specific insights.
- *Build Knowledge Graphs*: Construct knowledge graphs from responses for advanced analysis.
- *PDF Comparison*: Compare two PDFs to highlight differences and similarities.

## Technology Stack
- *Language*: Python
- *Core Library*: LangChain
- *Other Tools*: LangChain's detection models, Docker, OpenAI APIs

## Project Challenges
While LangChain served as the foundation, limitations in flexibility and scalability emerged as the features grew. This led to messy and unmanageable code. A structured approach using LangGraph could have improved workflow and efficiency, providing better modularity and readability.

## Directory Structure
```bash
AI_chats/
├── configuration/
│   ├── __init__.py
│   └── config.py
├── docker-compose.yml
├── Dockerfile
├── docker_rebuild.sh
├── main.py
├── openai_keys/
│   ├── __init__.py
│   └── openai_cred.json
├── output.pdf
├── README.md
├── requirements.txt
├── start.sh
├── tools/
│   ├── doc_chatting/
│   │   ├── __init__.py
│   │   ├── chat_with_document.py
│   │   ├── construct_knowledge_graph.py
│   │   ├── langchainVector.py
│   │   ├── link_scrapping_and_chating.py
│   │   ├── pdf_comparision_btw_two_files.py
│   │   ├── search_by_descrp_keyword.py
│   │   ├── table_image_extraction_pdf.py
│   │   └── talk_to_table.py
├── utils/
    ├── __init__.py
    ├── bucket.py
    ├── custom_logger.py
    ├── history.py
    ├── llm_cache.py
    ├── prompts.py
    └── utility.py
```

## Installation

### Clone the Repository
bash
git clone <repository-url>
cd AI_chats


### Install Dependencies
Use the provided requirements.txt file to install dependencies:
bash
pip install -r requirements.txt


### Set Up OpenAI API Keys
Add your OpenAI API keys to openai_keys/openai_cred.json.

### Build and Run Docker Containers
bash
docker-compose up --build


## Usage

### Start the Application
Run the main script to initiate the service:
bash
python main.py


### Access Features
- Upload documents via the interface.
- Use interactive Q&A to retrieve insights.
- Extract tables and compare PDFs directly.

### Customize Configuration
Modify configuration/config.py for custom settings.

## Development Workflow
- *Utilities*: Includes helper modules such as logging (custom_logger.py), history tracking (history.py), and prompt management (prompts.py).
- *Tools*: Implements core functionalities such as document chat, table extraction, and keyword search.
- *Configuration*: Handles application settings and parameters.

## Future Improvements
- *Switch to LangGraph*: Refactoring the project to use LangGraph for improved modularity.
- *Enhanced Flexibility*: Addressing limitations in LangChain for better scalability.
- *User Interface*: Adding a more intuitive and user-friendly front-end.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Special thanks to the contributors and the LangChain community for their support and resources.

---
Feel free to reach out for support or feature requests!
