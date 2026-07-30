from __future__ import annotations

from uuid import uuid4

import streamlit as st

from so_rag.config import Settings
from so_rag.hybrid_search import HybridSearchService
from so_rag.logging_setup import get_run_logger
from so_rag.models import HybridSearchHit
from so_rag.orchestrator import run_rag_pipeline
from so_rag.search import _parse_document


@st.cache_resource
def _hybrid_service(_settings: Settings) -> HybridSearchService:
    return HybridSearchService(_settings)


def _hit_body(hit: HybridSearchHit) -> str:
    body, _ = _parse_document(hit.document_text, hit.title, hit.tags)
    return body


def _render_ask_tab(settings: Settings) -> None:
    st.caption("LLM preprocessing, hybrid retrieval, reranking, and a grounded answer.")
    query = st.text_input(
        "Ask a question",
        placeholder="e.g. How do I sort a Python list in place?",
        key="ask_query",
    )
    technical = st.checkbox(
        "Technical details",
        value=False,
        help="Show RRF fusion and re-ranking scores.",
        key="ask_technical",
    )

    if st.button("Ask", type="primary", key="ask_btn") and query.strip():
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


def _render_search_tab(settings: Settings) -> None:
    st.caption(
        "Hybrid vector + BM25 search over the indexed corpus. "
        "No LLM — browse ranked posts directly (still capped by Top K)."
    )
    query = st.text_input(
        "Search query",
        placeholder="e.g. sort python list",
        key="search_query",
    )
    col1, col2, col3 = st.columns(3)
    top_k = col1.slider("Top K results", min_value=10, max_value=200, value=50, step=10, key="search_top_k")
    show_bodies = col2.checkbox("Show post bodies", value=False, key="search_bodies")
    technical = col3.checkbox("Show RRF scores", value=False, key="search_technical")

    if st.button("Search", type="primary", key="search_btn") and query.strip():
        with st.spinner("Searching..."):
            hits = _hybrid_service(settings).search(query.strip(), top_k=top_k)

        st.subheader(f"{len(hits)} results")
        if not hits:
            st.info("No indexed posts matched. Try a broader query or index more posts.")
            return

        for rank, hit in enumerate(hits, start=1):
            st.markdown(
                f"**{rank}. {hit.title}**  \n"
                f"Tags: `{hit.tags}`  \n"
                f"{hit.stackoverflow_url}"
            )
            if technical:
                parts = [f"rrf_score={hit.score:.6f}"]
                if hit.vector_rank is not None:
                    parts.append(f"vector_rank={hit.vector_rank}")
                if hit.bm25_rank is not None:
                    parts.append(f"bm25_rank={hit.bm25_rank}")
                if hit.rrf_vector_component is not None:
                    parts.append(f"vector_component={hit.rrf_vector_component:.6f}")
                if hit.rrf_lexical_component is not None:
                    parts.append(f"lexical_component={hit.rrf_lexical_component:.6f}")
                st.code("\n".join(parts), language="text")
            if show_bodies:
                body = _hit_body(hit)
                if body:
                    with st.expander("Body", expanded=False):
                        st.text(body)
            st.divider()


def main() -> None:
    st.set_page_config(page_title="Stack Overflow RAG", layout="wide")
    st.title("Stack Overflow RAG")
    settings = Settings()

    ask_tab, search_tab = st.tabs(["Ask (RAG)", "Search"])
    with ask_tab:
        _render_ask_tab(settings)
    with search_tab:
        _render_search_tab(settings)


if __name__ == "__main__":
    main()
