from __future__ import annotations

from uuid import uuid4

import streamlit as st

from so_rag.config import Settings
from so_rag.logging_setup import get_run_logger
from so_rag.orchestrator import run_rag_pipeline


def main() -> None:
    st.set_page_config(page_title="Stack Overflow RAG", layout="wide")
    st.title("Stack Overflow RAG")
    st.caption("Hybrid retrieval (vector + BM25), cross-encoder re-ranking, and grounded answers.")

    settings = Settings()
    query = st.text_input("Ask a question", placeholder="e.g. How do I sort a Python list in place?")
    technical = st.checkbox("Technical details", value=False, help="Show RRF fusion and re-ranking scores.")

    if st.button("Search", type="primary") and query.strip():
        run_id = str(uuid4())
        logger = get_run_logger(run_id, settings.log_dir)
        with st.spinner("Running pipeline..."):
            result = run_rag_pipeline(settings, query.strip(), run_id, logger)

        if result.llm_error:
            st.warning(result.llm_error)

        st.subheader("Answer")
        st.write(result.answer)

        st.subheader("Run")
        st.json(
            {
                "run_id": result.run_id,
                "latency_ms": result.latency_ms,
                "tags_extracted": result.tags_extracted,
                "rewritten_query": result.rewritten_query,
                "top_1_score": result.top_1_score,
                "context_source_ids": result.context_source_ids,
            }
        )

        with st.expander("Source posts", expanded=True):
            for src in result.sources:
                st.markdown(f"**{src.title}**  \nTags: `{src.tags}`  \n{src.stackoverflow_url}")
                if technical:
                    st.code(
                        f"rerank_score={src.rerank_score:.4f}\nrrf_score={src.rrf_score:.6f}",
                        language="text",
                    )


if __name__ == "__main__":
    main()
