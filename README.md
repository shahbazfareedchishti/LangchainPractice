# LangChain Playbook: RAG, Agents, and LLMs

This repository contains a collection of Python scripts demonstrating various concepts and implementations using the LangChain framework. It serves as a personal practice ground for exploring large language models (LLMs), prompt engineering, document retrieval, and complex agents.

## Project Structure

The project includes scripts that cover different aspects of LangChain, categorized as follows:

### Document Loading
Scripts for extracting text from different sources.
* `dataloader1.py`: Basic document loading.
* `directoryloader.py`: Loading multiple files from a directory.
* `pdfloader.py`: Retrieving and loading text from PDF documents.
* `webbaseloader.py`: Scraping and loading data from web pages.
* `WikipediaRetriver.py`: Fetching information from Wikipedia.

### Text Splitting & Processing
Scripts exploring ways to chunk large text for embedding and retrieval.
* `characterspliter.py`: Basic character-level splitting.
* `pythontextsplitter.py`: Splitting Python code intelligently.
* `recursivesplitter.py`: Advanced recursive text splitting.

### Prompts & Templates
Scripts demonstrating how to structure and format instructions for LLMs.
* `prompt.py`: Foundational prompt examples.
* `chatprompttemplate.py`: Creating templates specifically for chat models.
* `static_template.py`: Handling static prompt templates.
* `dynamic_template.py`: Creating and managing dynamic prompts.

### Vector Stores & Embeddings
Storing and retrieving text chunks efficiently using embeddings.
* `huggingfaceembeddings.py`: Generating text embeddings using HuggingFace models.
* `chromadbbbb.py`: Interacting with the Chroma vector database.
* `faisssss.py`: Implementing FAISS for similarity search.
* `vector_store_retriever.py`: Using vector stores as retrievers.
* `mmrr.py`: Maximum Marginal Relevance retrieval exploration.

### Output Parsers
Extracting structured information from model responses.
* `stroutputparser.py`: Simple string output parsing.
* `jsonoutputparser.py`: Forcing strict JSON output from LLMs.
* `pydanticoutputparser.py`: Validating LLM outputs using Pydantic schemas.
* `llm_pydant.py`: Further exploration of Pydantic and LLMs.

### Chains and Execution Logic
Combining multiple steps or executing conditional logic.
* `conditionalchains.py`: Running different paths based on LLM outputs or inputs.
* `parallelchain.py`: Executing multiple chains simultaneously.

### Retrieval-Augmented Generation (RAG)
Combining retrieved context with language models.
* `basic_rag.py`: A simple RAG implementation.
* `memoryrag.py`: RAG with conversational memory.
* `databasememoryrag.py`: Storing memory persistently using SQLite.
* `multidocumentrag.py`: Processing and retrieving over multiple documents.
* `explainable_rag.py`: A RAG pipeline that cites its sources.

### Agents & Tools
Giving language models the ability to execute code or call functions.
* `toolcallingagent.py`: An agent capable of determining when and how to call defined tools.

### Applications & Additional Concepts
* `bloggenerator.py`: A small application for generating blog content.
* `app.py` / `main.py`: Entry points or UI components.
* `annotatedtypedict.py` / `typeddict.py`: Exploration of Python typing in conjunction with LangChain.
* `judge.py`: Evaluation or judging script for LLM responses.

## Setup Instructions

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your environment variables (e.g., API keys) in a `.env` file.

## Usage

Most scripts can be executed directly as standalone programs:
```bash
python script_name.py
```
For applications utilizing Streamlit (if present), run:
```bash
streamlit run script_name.py
```

## Note

This codebase is primarily for learning and testing LangChain features.
