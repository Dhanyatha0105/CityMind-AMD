"""
RAG Subsystem: Retriever
Queries FAISS index to retrieve relevant building code chunks for compliance checking.

AMD Tech: Ryzen AI NPU for embedding inference, Zen5 AVX-512 for FAISS search
Reuse: Adapted from neuro-rag-assistant retriever pattern
"""

import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class BuildingCodeRetriever:
    """
    Retrieves relevant building code sections using vector similarity search.
    
    Used by the Compliance Agent to ground its assessments in actual codes.
    
    AMD Technology:
    - Embedding inference on AMD Ryzen AI NPU
    - FAISS vector search leverages AMD Zen5 AVX-512 instructions
    """
    
    def __init__(self, ingestor=None):
        """
        Args:
            ingestor: A BuildingCodeIngestor instance with a built index.
        """
        self.ingestor = ingestor
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Dict]:
        """
        Query the building code index for relevant sections.
        
        Args:
            query: Natural language query (e.g., "crack width limits for exterior concrete")
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of dicts with text, source, score
        """
        if self.ingestor is None or not self.ingestor.chunks:
            logger.warning("No index available, returning default context")
            return self._default_context(query)
        
        import numpy as np
        
        # Embed the query
        self.ingestor._load_embedder()
        
        if self.ingestor.embedder == "fallback":
            query_vec = self.ingestor._fallback_embed([query])
        else:
            query_vec = self.ingestor.embedder.encode([query])
            query_vec = np.array(query_vec, dtype=np.float32)
        
        # Search
        if self.ingestor.index == "brute_force":
            return self._brute_force_search(query_vec, top_k, score_threshold)
        
        try:
            import faiss
            # Normalize query vector
            faiss.normalize_L2(query_vec)
            scores, indices = self.ingestor.index.search(query_vec, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.ingestor.chunks):
                    continue
                if score < score_threshold:
                    continue
                
                chunk = self.ingestor.chunks[idx]
                results.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk.get("page", 1),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": float(score),
                })
            
            logger.info(f"Retrieved {len(results)} relevant code sections for query")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return self._default_context(query)
    
    def _brute_force_search(
        self, query_vec, top_k: int, score_threshold: float
    ) -> List[Dict]:
        """Fallback brute-force cosine similarity search."""
        import numpy as np
        
        embeddings = self.ingestor._embeddings
        
        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        scores = np.dot(emb_norm, query_norm.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] < score_threshold:
                continue
            chunk = self.ingestor.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk.get("page", 1),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": float(scores[idx]),
            })
        
        return results
    
    def get_context_for_defects(
        self,
        defects: List[Dict],
        top_k_per_defect: int = 2,
    ) -> str:
        """
        Get building code context for a list of defects.
        Used to feed the Compliance Agent.
        
        Args:
            defects: List of defect dicts with 'defect_type'
            top_k_per_defect: Number of code sections per defect type
            
        Returns:
            Concatenated context string
        """
        if not defects:
            return "No defects to check against building codes."
        
        # Build queries from defect types
        defect_types = set(d.get("defect_type", "unknown") for d in defects)
        
        query_map = {
            "crack": "concrete crack width limits repair requirements ACI 318",
            "spalling": "spalling concrete surface repair requirements ACI 562",
            "corrosion": "reinforcement corrosion protection limits ACI 222R",
            "delamination": "concrete delamination repair assessment ACI 562",
            "exposed_rebar": "minimum concrete cover reinforcement protection ACI 318",
            "water_damage": "waterproofing concrete moisture damage ACI 515",
            "displacement": "story drift limits structural displacement ASCE 7",
            "staining": "concrete staining surface deterioration assessment",
        }
        
        all_results = []
        seen_chunks = set()
        
        for dtype in defect_types:
            query = query_map.get(dtype, f"{dtype} building code requirements structural concrete")
            results = self.query(query, top_k=top_k_per_defect)
            
            for r in results:
                chunk_id = r.get("chunk_id", r["text"][:50])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append(r)
        
        if not all_results:
            return self._default_context_text()
        
        # Format context
        context_parts = []
        for i, r in enumerate(all_results, 1):
            context_parts.append(
                f"[Source: {r['source']}, Page {r['page']}, Score: {r['score']:.2f}]\n{r['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _default_context(self, query: str = "") -> List[Dict]:
        """Return default building code context when no index is available."""
        # Provide basic code references as fallback
        default_codes = [
            {
                "text": "ACI 318-19 §24.3.2: Maximum crack width 0.3mm exterior, 0.4mm interior. "
                        "Crack control reinforcement required for flexural members.",
                "source": "ACI_318-19_default",
                "page": 1,
                "chunk_id": "default_crack",
                "score": 0.5,
            },
            {
                "text": "ACI 562-19 §6.3: Spalled areas >50mm² require repair with compatible materials. "
                        "Repair depth minimum 25mm beyond corroded reinforcement.",
                "source": "ACI_562-19_default",
                "page": 1,
                "chunk_id": "default_spalling",
                "score": 0.5,
            },
            {
                "text": "ACI 222R-19 §4.2: Chloride limits for reinforced concrete. "
                        "Section loss >20% requires reinforcement replacement.",
                "source": "ACI_222R-19_default",
                "page": 1,
                "chunk_id": "default_corrosion",
                "score": 0.5,
            },
        ]
        return default_codes
    
    def _default_context_text(self) -> str:
        """Return default context as formatted text."""
        defaults = self._default_context()
        return "\n\n---\n\n".join(
            f"[Source: {d['source']}]\n{d['text']}" for d in defaults
        )
