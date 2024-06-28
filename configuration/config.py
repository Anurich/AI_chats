file_config = {
    "web-loader":{
        "chunk_size": 200,
        "chunk_overlap": 100,
        "persist_directory": "web_loader_chromadb",
        "urls":  ["https://www.espn.com/", "https://google.com"]
    },
    "chat_with_pdf":{
        "chunk_size": 2000,
        "chunk_overlap": 500,
        "persist_directory": "chroma_store",
        "filenames":["http://162.243.160.161/salon/ai/Priya-resume.pdf"],
        "filename_table":[]
    }
}