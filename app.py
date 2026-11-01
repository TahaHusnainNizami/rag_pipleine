
import hashlib

import chromadb
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from chat2 import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    chat_with_llm,
    email_agent,
    is_email_request,
)

load_dotenv()

st.set_page_config(
    page_title="Ragify + Email Generator",
    page_icon="📚",
    layout="centered"
)

st.markdown(
    """
    <style>
    .email-agent-card {
        padding: 1.25rem;
        border: 1px solid rgba(49, 51, 63, .18);
        border-radius: .75rem;
        background: linear-gradient(135deg, #f8f9ff, #ffffff);
        margin: .5rem 0 1rem;
    }
    .email-agent-card h2 {
        margin: 0;
        color: #273469;
    }
    .email-agent-card p {
        color: #62677a;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Ragify + Email Generator")
st.caption("Upload your PDFs and chat with your documents.")


# ---------------- DATABASE ----------------

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------- TEXT CHUNKING ----------------

def split_text(text, chunk_size=1000, overlap=150):

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# ---------------- PDF PROCESSING ----------------

def process_pdf(uploaded_file, collection, embedding_model):

    reader = PdfReader(uploaded_file)

    documents = []
    metadatas = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        chunks = split_text(text)

        for chunk_number, chunk in enumerate(chunks, start=1):

            documents.append(chunk)

            metadatas.append({
                "source": uploaded_file.name,
                "page": page_number,
                "chunk": chunk_number
            })

    if not documents:
        return 0, "No readable text found in this PDF."

    # Generate embeddings
    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    # Unique IDs for uploaded chunks
    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

    ids = [
        f"upload-{file_hash}-{i}"
        for i in range(len(documents))
    ]

    # Upsert makes re-uploading the same file safe and prevents duplicate chunks.
    for metadata in metadatas:
        metadata["file_hash"] = file_hash

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents), None


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_sources" not in st.session_state:
    st.session_state.uploaded_sources = set()

if "email_requested" not in st.session_state:
    st.session_state.email_requested = False

if "email_draft" not in st.session_state:
    st.session_state.email_draft = None


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Add to Knowledge Base",
        disabled=not uploaded_files,
        use_container_width=True
    ):

        try:
            collection = load_collection()
            embedding_model = load_embedding_model()

            for pdf_file in uploaded_files:

                with st.spinner(
                    f"Processing {pdf_file.name}..."
                ):

                    count, error = process_pdf(
                        pdf_file,
                        collection,
                        embedding_model
                    )

                if error:
                    st.warning(
                        f"{pdf_file.name}: {error}"
                    )
                else:
                    st.session_state.uploaded_sources.add(pdf_file.name)
                    st.success(
                        f"{pdf_file.name}: {count} chunks added."
                    )

        except Exception as e:
            st.error("Error processing PDF.")
            st.exception(e)

    st.divider()

    st.header("✉️ Email Agent")
    st.caption("Create, review, and send emails with AI assistance.")
    if st.button(
        "Open Email Agent",
        type="primary",
        use_container_width=True,
        key="open_email_agent",
    ):
        st.session_state.email_requested = True
        st.session_state.email_draft = None
        st.rerun()

    st.divider()

    st.header("⚙️ Settings")

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.chat_history = []

        st.rerun()


# ---------------- LOAD RAG COMPONENTS ----------------

try:

    collection = load_collection()
    embedding_model = load_embedding_model()

except Exception as e:

    st.error(
        "Unable to connect to ChromaDB. "
        "Check your database path and collection name in chat2.py."
    )

    st.exception(e)
    st.stop()


st.caption(
    f"Knowledge Base: {collection.count():,} chunks"
)


# ---------------- CHAT INTERFACE ----------------

with st.expander("How to use"):

    st.markdown(
        """
        1. Upload one or more PDF files from the sidebar.
        2. Click Add to Knowledge Base.
        3. Wait for document processing to finish.
        4. Ask questions about your documents.
        5. Share information and ask follow-up questions.

        Conversation history is temporary and lasts for this session.
        """
    )


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# User input

prompt = st.chat_input(
    "Ask something about your documents..."
)


if prompt:

    # Display user message

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    if is_email_request(prompt):
        st.session_state.email_requested = True
        st.rerun()

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer = chat_with_llm(
                    question=prompt,
                    collection=collection,
                    embedding_model=embedding_model,
                    history=st.session_state.chat_history,
                    n_results=5,
                    preferred_sources=st.session_state.uploaded_sources,
                )
                st.markdown(answer)
            except Exception as e:
                answer = "Sorry, something went wrong."
                st.error(answer)
                st.exception(e)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
    st.session_state.chat_history.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])


if st.session_state.email_requested:
    st.markdown(
        """
        <div class="email-agent-card">
            <h2>✉️ Email Agent</h2>
            <p>Tell the agent who to email and what you want to say. You will
            review the generated draft before anything is sent.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("email_details"):
        recipient = st.text_input(
            "Recipient email",
            placeholder="recipient@example.com",
        )
        purpose = st.text_area(
            "What should this email be about?",
            placeholder="Describe the purpose of the email...",
            height=120,
        )
        tone = st.selectbox(
            "Tone",
            ["professional", "friendly", "formal", "concise"],
        )
        key_points = st.text_input(
            "Key points",
            placeholder="Optional: deadline, attachment, next steps",
        )
        attachments = st.file_uploader(
            "Attachments",
            accept_multiple_files=True,
            help="Attach files to include with the email. Maximum 25 MB per file.",
        )
        draft_col, cancel_col = st.columns(2)
        with draft_col:
            create_draft = st.form_submit_button(
                "✨ Generate draft",
                type="primary",
                use_container_width=True,
            )
        with cancel_col:
            cancel_agent = st.form_submit_button(
                "Cancel",
                use_container_width=True,
            )

    if cancel_agent:
        st.session_state.email_requested = False
        st.rerun()

    if create_draft:
        oversized = [
            file.name for file in attachments
            if file.size > 25 * 1024 * 1024
        ]
        if oversized:
            st.error(
                "These attachments exceed the 25 MB per-file limit: "
                + ", ".join(oversized)
            )
            st.stop()
        result = email_agent({
            "recipient": recipient,
            "purpose": purpose,
            "tone": tone,
            "key_points": [
                point.strip()
                for point in key_points.split(",")
                if point.strip()
            ],
        })
        if result.get("status") == "confirmation_required":
            result["attachments"] = [
                {
                    "filename": file.name,
                    "content": file.getvalue(),
                    "content_type": file.type,
                }
                for file in attachments
            ]
            st.session_state.email_draft = result
            st.session_state.email_requested = False
            st.rerun()
        st.error(result.get("error", "Could not create the email draft."))


if st.session_state.email_draft:
    draft = st.session_state.email_draft
    st.markdown(
        """
        <div class="email-agent-card">
            <h2>📝 Review email draft</h2>
            <p>Check the content carefully before sending.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(f"**To:** {draft['recipient']}")
    st.write(f"**Subject:** {draft['subject']}")
    if draft.get("attachments"):
        st.write(
            "**Attachments:** "
            + ", ".join(
                attachment["filename"] for attachment in draft["attachments"]
            )
        )
    st.markdown(draft["body"])
    col_send, col_cancel = st.columns(2)
    with col_send:
        if st.button("Send email", type="primary"):
            attachment_summary = [
                f"{attachment['filename']} "
                f"({len(attachment['content']):,} bytes)"
                for attachment in draft.get("attachments", [])
            ]
            result = email_agent({
                "recipient": draft["recipient"],
                "subject": draft["subject"],
                "body": draft["body"],
                "attachments": draft.get("attachments", []),
                "confirmed": True,
            })
            if result.get("status") == "sent":
                if attachment_summary:
                    st.success(
                        f"Email sent to {result['recipient']} with "
                        f"{', '.join(attachment_summary)}."
                    )
                else:
                    st.success(f"Email sent to {result['recipient']}.")
                st.session_state.email_draft = None
                st.rerun()
            st.error(result.get("error", "Email could not be sent."))
    with col_cancel:
        if st.button("Cancel email"):
            st.session_state.email_draft = None
            st.rerun()
