import os
import tempfile
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, List, Optional, TypedDict

from dotenv import load_dotenv
import streamlit as st
import torch
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType, DoclingLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

load_dotenv(Path(__file__).resolve().parent / ".env")

# ===============================
# SECTION 1: Contract Analysis Workflow
# ===============================
EMBED_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
# HF serverless chat model that works with current Inference providers.
GEN_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
EXPORT_TYPE = ExportType.DOC_CHUNKS
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or ""

_PDF_PIPELINE = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    force_backend_text=True,
)
_DOC_CONVERTER = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_PDF_PIPELINE)}
)


class HFChatLLM(LLM):
    """LangChain LLM wrapper around Hugging Face chat completions."""

    model_id: str = GEN_MODEL_ID
    max_new_tokens: int = 768
    huggingfacehub_api_token: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "hf_chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        token = self.huggingfacehub_api_token or HF_TOKEN
        if not token:
            raise RuntimeError("HF_TOKEN is missing. Add it to your .env file.")
        client = InferenceClient(token=token)
        response = client.chat_completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
        )
        content = response.choices[0].message.content
        return content or ""


def get_llm(max_new_tokens: int = 768) -> HFChatLLM:
    return HFChatLLM(
        model_id=GEN_MODEL_ID,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=HF_TOKEN,
    )


class State(TypedDict):
    pdf_path: str
    chunks: list
    vectorstore: Milvus
    key_clauses: str
    risk_assessment: dict
    risk_assessment_report: str
    recommended_actions: str
    contract_text: str


def extract_chunks(state: State):
    """Extracts chunks from the PDF."""
    loader = DoclingLoader(
        file_path=state["pdf_path"],
        converter=_DOC_CONVERTER,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
    if not splits:
        raise ValueError("No text could be extracted from the PDF.")
    state["chunks"] = splits
    return state


def embedding_and_store(state: State):
    """Embeds chunks and stores them in Milvus."""
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    splits = [Document(page_content=split) for split in state["chunks"]]
    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name="docling_demo",
        connection_args={"uri": MILVUS_URI},
        index_params={"index_type": "FLAT", "metric_type": "L2", "params": {}},
        drop_old=True,
    )
    state["vectorstore"] = vectorstore
    return state


def key_clauses(state: State):
    """Retrieves relevant documents and extracts key clauses."""
    retriever = state["vectorstore"].as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke("Extract key clauses from the contract")
    contract_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    state["contract_text"] = contract_text

    prompt = PromptTemplate(
        input_variables=["context"],
        template="""You are an AI legal assistant with expertise in contract analysis. Your task is to extract and summarize key clauses from the following contract.
Do not include irrelevant information except clauses from document.
**Instructions**:
- Identify all relevant clauses related to liabilities, risks, obligations, and rights.
- Provide a concise yet comprehensive summary of each clause.
- Retain legal accuracy while simplifying complex legal language.

**Example Format**:
- **Clause Type**: [Liability, Termination, Confidentiality, Risk, etc.]
- **Extracted Clause**: [Verbatim or Summarized]
- **Summary**: [Briefly explain in simple terms]

**Contract Text**:
{context}

**Extracted Clauses & Summaries**:
""",
    )
    llm = get_llm(max_new_tokens=900)
    state["key_clauses"] = llm.invoke(prompt.format(context=contract_text))
    return state


def generate_risk_assessment_report(state: State):
    """Generates a risk assessment report."""
    contract_text = state.get("contract_text", "")
    report_prompt = f"""You are an expert in legal risk assessment. Analyze the contract below and identify potential risks.

**Instructions**:
- Identify and classify risks (e.g., financial, legal, compliance, reputational).
- Assess severity (Low, Medium, High) and likelihood (Unlikely, Possible, Likely).
- Explain potential consequences of each risk.

**Example Format**:
- **Risk Type**: [Legal, Compliance, Financial, Reputational, Operational]
- **Clause Reference**: [If applicable, mention specific clauses]
- **Risk Level**: [Low, Medium, High]
- **Likelihood**: [Unlikely, Possible, Likely]
- **Potential Consequence**: [Briefly describe impact]

**Contract Text**:
{contract_text}

**Risk Assessment Report**:
"""
    llm = get_llm(max_new_tokens=900)
    state["risk_assessment_report"] = llm.invoke(report_prompt)
    return state


def recommend_actions(state: State):
    """Recommends actions based on the risk assessment report."""
    report = state["risk_assessment_report"]
    action_prompt = f"""You are a legal expert. Analyze the following risk assessment report and suggest specific actions for each clause:

Risk Assessment Report:
{report}

Provide your recommendations in the following format:

**Clause**: [Clause Text]
**Risk Level**: [Risk Level]
**Recommended Action**: [Specific Action]
"""
    llm = get_llm(max_new_tokens=900)
    state["recommended_actions"] = llm.invoke(action_prompt)
    return state


graph = StateGraph(State)
graph.add_node("extract_chunks", extract_chunks)
graph.add_node("embedding_and_store", embedding_and_store)
graph.add_node("Clauses", key_clauses)
graph.add_node("generate_report", generate_risk_assessment_report)
graph.add_node("recommend_actions", recommend_actions)

graph.set_entry_point("extract_chunks")
graph.add_edge("extract_chunks", "embedding_and_store")
graph.add_edge("embedding_and_store", "Clauses")
graph.add_edge("Clauses", "generate_report")
graph.add_edge("generate_report", "recommend_actions")

risk_graph = graph.compile()

# ===============================
# SECTION 2: Fine-tuned Chatbot Setup for Conversational RAG
# ===============================
CHAT_MODEL_NAME = "ramonad2024/contract_lora_model"
_chat_model = None
_chat_tokenizer = None
_chat_load_error = None


def _ensure_chat_model():
    """Load the fine-tuned chatbot on first use."""
    global _chat_model, _chat_tokenizer, _chat_load_error
    if _chat_model is not None:
        return _chat_model, _chat_tokenizer
    if _chat_load_error is not None:
        raise RuntimeError(_chat_load_error)
    try:
        from transformers import BitsAndBytesConfig
        from unsloth import FastLanguageModel

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            CHAT_MODEL_NAME,
            max_seq_length=2048,
            dtype=torch.float16,
            quantization_config=quantization_config,
        )
        FastLanguageModel.for_inference(model)
        _chat_model, _chat_tokenizer = model, tokenizer
        return _chat_model, _chat_tokenizer
    except Exception as exc:
        _chat_load_error = (
            f"Chatbot model unavailable on this machine ({exc}). "
            "Contract analysis still works if HF_TOKEN is set."
        )
        raise RuntimeError(_chat_load_error) from exc


def _build_contract_context(analysis: dict | None) -> str:
    """Build a compact context string from analysis output for chat."""
    if not analysis:
        return ""
    parts = []
    if analysis.get("contract_text"):
        parts.append(f"Retrieved contract excerpts:\n{analysis['contract_text']}")
    if analysis.get("key_clauses"):
        parts.append(f"Extracted clauses:\n{analysis['key_clauses']}")
    if analysis.get("risk_assessment_report"):
        parts.append(f"Risk assessment:\n{analysis['risk_assessment_report']}")
    if analysis.get("recommended_actions"):
        parts.append(f"Recommended actions:\n{analysis['recommended_actions']}")
    # Keep prompt size manageable for the inference API.
    context = "\n\n".join(parts)
    return context[:12000]


def generate_response(prompt: str, analysis: dict | None = None) -> str:
    """Answer using uploaded-contract analysis context when available."""
    context = _build_contract_context(analysis)
    if context:
        chat_prompt = f"""You are a helpful legal assistant. Answer ONLY using the contract analysis context below.
If the answer is not in the context, say what is missing. Be brief and clear.
Do not claim you cannot access the uploaded contract — the context is provided.

Contract analysis context:
{context}

User question: {prompt}

Answer:"""
        return get_llm(max_new_tokens=400).invoke(chat_prompt)

    # No uploaded analysis yet — try local fine-tuned model, then generic HF chat.
    try:
        model, tokenizer = _ensure_chat_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            top_p=0.9,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception:
        return get_llm(max_new_tokens=256).invoke(
            "You are a helpful legal assistant. Answer briefly and clearly.\n\n"
            f"Question: {prompt}\n\nAnswer:"
        )


# ===============================
# SECTION 3: Streamlit UI
# ===============================
st.title("Contract Risk Assessment")

if not HF_TOKEN:
    st.warning("HF_TOKEN is not set. Add it to `.env` before analyzing contracts.")

uploaded_file = st.file_uploader("Upload a PDF contract", type="pdf")

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"
    tab1, tab2 = st.tabs(["Analyze Contract", "Legal Assistant"])

    with tab1:
        if (
            st.session_state.get("analysis_file_key") != file_key
            or "analysis_output" not in st.session_state
        ):
            with st.spinner("Analyzing contract..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(uploaded_file.getbuffer())
                    temp_pdf_path = temp_pdf.name
                try:
                    output = risk_graph.invoke(input={"pdf_path": temp_pdf_path})
                    st.session_state.analysis_output = output
                    st.session_state.analysis_file_key = file_key
                    st.session_state.messages = []
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.stop()
                finally:
                    try:
                        os.unlink(temp_pdf_path)
                    except OSError:
                        pass

        output = st.session_state.analysis_output
        selected_tab = st.selectbox(
            "Select a Section",
            ["Extracted Clauses", "Risk Assessment Report", "Recommended Actions"],
        )
        if selected_tab == "Extracted Clauses":
            st.markdown(output.get("key_clauses") or "_No clauses extracted._")
        elif selected_tab == "Risk Assessment Report":
            st.write(output.get("risk_assessment_report") or "_No report generated._")
        elif selected_tab == "Recommended Actions":
            st.write(output.get("recommended_actions") or "_No actions generated._")

    with tab2:
        st.header("Chat with the Legal Assistant")
        st.caption("Answers use your uploaded contract analysis from the Analyze tab.")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render history first so chat_input stays at the bottom.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask me anything about this contract...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Generating response..."):
                try:
                    response = generate_response(
                        user_input,
                        analysis=st.session_state.get("analysis_output"),
                    )
                except Exception as exc:
                    response = f"Chatbot unavailable: {exc}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
