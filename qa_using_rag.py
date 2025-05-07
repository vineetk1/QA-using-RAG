import os
from google import genai
from google.genai import types
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Provide API-Key; for security reasons there are other ways of providing API-Key
# but those are not used here
API_KEY = <API_KEY> 
client = genai.Client(api_key=API_KEY)

#for m in client.models.list():
#    if "embedContent" in m.supported_actions:
#        print(m.name)

DOC1 = "While The Python Language Reference describes the exact syntax and semantics of the Python language, this library reference manual describes the standard library that is distributed with Python. It also describes some of the optional components that are commonly included in Python distributions."
DOC2 = "Python’s standard library is very extensive, offering a wide range of facilities as indicated by the long table of contents listed below. The library contains built-in modules (written in C) that provide access to system functionality such as file I/O that would otherwise be inaccessible to Python programmers, as well as modules written in Python that provide standardized solutions for many problems that occur in everyday programming. Some of these modules are explicitly designed to encourage and enhance the portability of Python programs by abstracting away platform-specifics into platform-neutral APIs."
DOC3 = "The Python installers for the Windows platform usually include the entire standard library and often also include many additional components. For Unix-like operating systems Python is normally provided as a collection of packages, so it may be necessary to use the packaging tools provided with the operating system to obtain some or all of the optional components."
DOC4 = "Python’s standard library is very extensive, offering a wide range of facilities as indicated by the long table of contents listed below. The library contains built-in modules (written in C) that provide access to system functionality such as file I/O that would otherwise be inaccessible to Python programmers, as well as modules written in Python that provide standardized solutions for many problems that occur in everyday programming. Some of these modules are explicitly designed to encourage and enhance the portability of Python programs by abstracting away platform-specifics into platform-neutral APIs."
docs = [DOC1, DOC2, DOC3, DOC4]

class embedding_fn(EmbeddingFunction):
    doc_mode = True

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embed_task = "retrieval_document" if self.doc_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embed_task,
            ),
        )
        return [emb.values for emb in response.embeddings]

print("NOTE: In the following Answer, Passages are not part of the input to the LLM")
query = "language used in python library"
#prompt = f"""Answer the question in a conversational tone.
prompt = f"""Answer the question in an easy-to-read but concise format. Remove empty newlines.
QUESTION: {query.replace("\n", " ")}"""
answer = client.models.generate_content(
    model='gemma-3-27b-it',
    contents=prompt)
print(f"PROMPT: {prompt}")
print(f"ANSWER (Passages are not part of the input to the LLM):\n {answer.text}")

DB_NAME = "googlecardb"
embed_fn = embedding_fn()
embed_fn.doc_mode = True
chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
db.add(documents=docs, ids=[str(i) for i in range(len(docs))])
#print(db.count())
#print(db.peek(1))
embed_fn.doc_mode = False
result = db.query(query_texts=[query], n_results=10)

print("NOTE: In the following Answer, Passages are also part of the input to the LLM")
[passages] = result["documents"]
for passage in passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"
print(f"PROMPT: {prompt}")
answer = client.models.generate_content(
    model='gemma-3-27b-it',
    contents=prompt)
print(f"ANSWER (Passages are also part of the input to the LLM):\n {answer.text}")
