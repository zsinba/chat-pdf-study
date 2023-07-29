import pdfplumber
import PyPDF4
import re
import os
import sys
from typing import Callable, List, Tuple, Dict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv



"""
这个脚本可以处理PDF文档，并使用OpenAI的GPT-3.5架构生成文档嵌入。以下是每个函数的简要概述：

extract_metadata_from_pdf(file_path: str) -> dict：该函数从PDF文件中提取元数据，例如标题、作者和创建日期，并将其作为字典返回。

extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]：该函数提取PDF每一页的文本，并将其作为包含页面编号和提取文本的元组列表返回。

parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]：该函数结合前两个函数，提取PDF文档的元数据和文本，并返回一个包含提取文本和元数据的元组。

merge_hyphenated_words(text: str) -> str：该函数将跨行分割的连字符单词替换为完整单词。

fix_newlines(text: str) -> str：该函数将单个换行符替换为空格。

remove_multiple_newlines(text: str) -> str：该函数删除多个连续换行符，并将它们替换为单个换行符。

clean_text(pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]) -> List[Tuple[int, str]]：该函数将一系列的清洗函数应用于每一页提取的文本，并返回一个清洗后的元组列表，其中包含页面编号和清理后的文本。

text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]：该函数将清洗后的包含页面编号和文本的元组列表转换为Document对象列表，这些对象将用于生成嵌入。

if __name__ == "__main__":：这是脚本的主要入口点。它加载PDF文件，将清洗函数应用于提取的文本，将清洗后的文本转换为Document对象列表，使用OpenAI的GPT-3.5架构为Document对象生成嵌入，并将这些嵌入存储在Chroma向量存储中。最后，它将向量存储保存在本地。
"""


'''
该函数从PDF文件中提取元数据，例如标题、作者和创建日期，并将其作为字典返回。
'''


def extract_metadata_from_pdf(file_path: str) -> dict:
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)  # Change this line
        metadata = reader.getDocumentInfo()
        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }


'''
该函数提取PDF每一页的文本，并将其作为包含页面编号和提取文本的元组列表返回。
'''


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages


'''
该函数结合前两个函数，提取PDF文档的元数据和文本，并返回一个包含提取文本和元数据的元组
'''


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


'''
该函数将跨行分割的连字符单词替换为完整单词。
'''


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


'''
该函数将单个换行符替换为空格
'''


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


"""
该函数删除多个连续换行符，并将它们替换为单个换行符。
"""


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


'''
该函数将一系列的清洗函数应用于每一页提取的文本，并返回一个清洗后的元组列表，其中包含页面编号和清理后的文本。
'''


def clean_text(
        pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


'''
该函数将清洗后的包含页面编号和文本的元组列表转换为Document对象列表，这些对象将用于生成嵌入。
'''


def text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []

    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


if __name__ == "__main__":
    load_dotenv()

    # Step 1: Parse PDF
    file_path = "/Users/sbin/Downloads/chat-pdf-study/src/data/april-2023.pdf"
    raw_pages, metadata = parse_pdf(file_path)

    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    # Optional: Reduce embedding cost by only using the first 23 pages
    document_chunks = document_chunks[:70]

    # Step 3 + 4: Generate embeddings and store them in DB
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name="april-2023-economic",
        persist_directory="src/data/chroma",
    )

    # Save DB locally
    vector_store.persist()
