from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.9)
else:
    device = torch.device("cpu")
    print("Warning: Running on CPU. Performance might be affected.")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
prompt_template = "Context:\n{context}\nQuestion: {question}\nAnswer:"
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
llm = HuggingFacePipeline(pipeline=text_generator)
llm_chain = LLMChain(llm=llm, prompt=PROMPT)
try:
    text_loader = TextLoader("output.txt")
except FileNotFoundError:
    print("Please create a 'output.txt' file with your Q&A dataset in the current directory.")
    exit()

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
documents = text_loader.load()
texts = text_splitter.split_documents(documents)
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)
def python_chatbot_rag(user_input, task):
    if task == "Error Correction":
        prompt = f"Correct the following Python code and explain the correction:\n{user_input}\nCorrected Code and Explanation:"
    elif task == "Q&A":
        return qa_chain.run(user_input)
    elif task == "Code Generation":
        prompt = f"Generate Python code for the following request:\n{user_input}\nGenerated Python Code:"
    else:
        return "Invalid task selected."    
    response = text_generator(prompt, max_length=500, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=0.7, top_p=0.9)[0]["generated_text"]
    return response[len(prompt):].strip()
def gradio_interface(user_input, task):
    if not user_input.strip():
        return "Please enter some input."
    return python_chatbot_rag(user_input, task)
with gr.Blocks() as python_chat_ui:
    gr.Markdown("# JEBS PYTHON CB")
    gr.Markdown("This chatbot helps with Python error correction, Q&A, and code generation using RAG.")
    user_input = gr.Textbox(lines=5, placeholder="Enter your Python code or question here...", label="Your Input")
    task = gr.Radio(["Error Correction", "Q&A", "Code Generation"], label="Select Task")
    output = gr.Textbox(lines=10, label="Chatbot Response")
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear")
    submit_button.click(gradio_interface, inputs=[user_input, task], outputs=output)
    clear_button.click(lambda: ("", ""), None, [user_input, output])
python_chat_ui.launch()
