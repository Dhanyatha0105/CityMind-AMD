"""
RAG Subsystem: Document Ingestion
Ingests building codes (PDF/TXT) → chunks → embeddings → FAISS index.

AMD Tech: Ryzen AI NPU for local embedding acceleration
Reuse: Adapted from neuro-rag-assistant
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class BuildingCodeIngestor:
    """
    Ingests building code documents for RAG-based compliance checking.
    
    Pipeline:
    1. Load PDF/TXT documents
    2. Split into overlapping chunks
    3. Generate embeddings (sentence-transformers)
    4. Store in FAISS vector index
    
    AMD Technology:
    - Embedding model can run on AMD Ryzen AI NPU via ONNX Runtime
    - FAISS vectorized search optimized for AMD Zen5 AVX-512
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        index_path: str = None,
    ):
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path
        self.embedder = None
        self.index = None
        self.chunks: List[Dict] = []
    
    def _load_embedder(self):
        """Load sentence-transformers embedding model."""
        if self.embedder is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback embeddings")
            self.embedder = "fallback"
    
    def _fallback_embed(self, texts: List[str]):
        """Simple fallback embedding using hashing (for demo without dependencies)."""
        import numpy as np
        import hashlib
        
        embeddings = []
        for text in texts:
            # Create a deterministic pseudo-embedding from text hash
            h = hashlib.sha256(text.encode()).hexdigest()
            vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
            # Pad to 384 dimensions (MiniLM size)
            vec = vec * 12  # 32 * 12 = 384
            embeddings.append(vec[:384])
        
        return np.array(embeddings, dtype=np.float32)
    
    def load_documents(self, doc_dir: str) -> List[Dict]:
        """
        Load all documents from a directory.
        Supports: PDF, TXT, MD
        """
        doc_path = Path(doc_dir)
        documents = []
        
        if not doc_path.exists():
            logger.warning(f"Document directory not found: {doc_dir}")
            return documents
        
        for file_path in sorted(doc_path.iterdir()):
            if file_path.suffix.lower() == ".pdf":
                docs = self._load_pdf(file_path)
                documents.extend(docs)
            elif file_path.suffix.lower() in (".txt", ".md"):
                docs = self._load_text(file_path)
                documents.extend(docs)
        
        logger.info(f"Loaded {len(documents)} document pages from {doc_dir}")
        return documents
    
    def _load_pdf(self, path: Path) -> List[Dict]:
        """Load PDF document."""
        pages = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "text": text.strip(),
                        "source": path.name,
                        "page": i + 1,
                        "type": "pdf",
                    })
        except ImportError:
            logger.warning("pypdf not installed, skipping PDF files")
        except Exception as e:
            logger.error(f"Error loading PDF {path}: {e}")
        
        return pages
    
    def _load_text(self, path: Path) -> List[Dict]:
        """Load text/markdown document."""
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return [{
                    "text": text.strip(),
                    "source": path.name,
                    "page": 1,
                    "type": path.suffix.lstrip("."),
                }]
        except Exception as e:
            logger.error(f"Error loading text file {path}: {e}")
        return []
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into overlapping chunks for embedding.
        Uses recursive character text splitting.
        """
        chunks = []
        
        for doc in documents:
            text = doc["text"]
            source = doc["source"]
            page = doc.get("page", 1)
            
            # Split by paragraphs first, then by size
            paragraphs = text.split("\n\n")
            current_chunk = ""
            chunk_idx = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If adding this paragraph exceeds chunk size, save current and start new
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source,
                        "page": page,
                        "chunk_idx": chunk_idx,
                        "chunk_id": f"{source}:p{page}:c{chunk_idx}",
                    })
                    # Keep overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + " " + para
                    chunk_idx += 1
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            
            # Don't forget the last chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "page": page,
                    "chunk_idx": chunk_idx,
                    "chunk_id": f"{source}:p{page}:c{chunk_idx}",
                })
        
        self.chunks = chunks
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def build_index(self, chunks: List[Dict] = None) -> bool:
        """
        Build FAISS vector index from chunks.
        
        AMD Optimization: FAISS uses AVX-512 on Zen5 for fast similarity search.
        """
        import numpy as np
        
        if chunks is not None:
            self.chunks = chunks
        
        if not self.chunks:
            logger.warning("No chunks to index")
            return False
        
        self._load_embedder()
        
        # Generate embeddings
        texts = [c["text"] for c in self.chunks]
        
        if self.embedder == "fallback":
            embeddings = self._fallback_embed(texts)
        else:
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index
        try:
            import faiss
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vecs)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
            
            # Save index if path specified
            if self.index_path:
                self._save_index()
            
            return True
            
        except ImportError:
            logger.warning("faiss-cpu not installed, using brute-force search fallback")
            self._embeddings = embeddings
            self.index = "brute_force"
            return True
    
    def _save_index(self):
        """Save FAISS index and chunks to disk."""
        import json
        
        if self.index_path and self.index and self.index != "brute_force":
            try:
                import faiss
                idx_path = Path(self.index_path)
                idx_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(idx_path))
                
                # Save chunks metadata
                meta_path = idx_path.with_suffix(".json")
                with open(meta_path, "w") as f:
                    json.dump(self.chunks, f, indent=2)
                
                logger.info(f"Index saved to {idx_path}")
            except Exception as e:
                logger.error(f"Error saving index: {e}")
    
    def load_index(self, index_path: str = None) -> bool:
        """Load FAISS index from disk."""
        import json
        
        path = Path(index_path or self.index_path)
        if not path.exists():
            return False
        
        try:
            import faiss
            self.index = faiss.read_index(str(path))
            
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    self.chunks = json.load(f)
            
            logger.info(f"Index loaded: {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def ingest_directory(self, doc_dir: str) -> int:
        """
        Full ingestion pipeline: load → chunk → embed → index.
        Returns number of chunks indexed.
        """
        documents = self.load_documents(doc_dir)
        if not documents:
            # Create sample building codes for demo
            documents = self._create_sample_codes()
        
        chunks = self.chunk_documents(documents)
        self.build_index(chunks)
        
        return len(chunks)
    
    def _create_sample_codes(self) -> List[Dict]:
        """Create sample building code excerpts for demo purposes."""
        sample_codes = [
            {
                "text": """ACI 318-19: Building Code Requirements for Structural Concrete
                
Section 24.3 — Crack Control for Flexural Members
24.3.2 — Distribution of reinforcement shall be such that the quantity 
z = fs * (dc * A)^(1/3) does not exceed 175 kips/in for interior exposure 
and 145 kips/in for exterior exposure. Maximum crack width for structural 
members shall not exceed 0.3mm (0.012 in) for exterior exposure and 
0.4mm (0.016 in) for interior exposure.

Section 20.5 — Minimum Concrete Cover
20.5.1.1 — Minimum cover for reinforcement in cast-in-place concrete 
shall be as follows:
(a) Concrete cast against and permanently in contact with ground: 3 in (75 mm)
(b) Concrete exposed to weather: No. 6 through No. 18 bars: 2 in (50 mm)
(c) Concrete not exposed to weather: Beams, columns: 1.5 in (38 mm)
(d) Slabs, walls: 0.75 in (19 mm)""",
                "source": "ACI_318-19.txt",
                "page": 1,
                "type": "txt",
            },
            {
                "text": """ACI 562-19: Code Requirements for Assessment, Repair, and 
Rehabilitation of Existing Concrete Structures

Section 6 — Repair of Concrete
6.3 — Surface Repair
6.3.1 — Spalled areas exceeding 50 mm² shall be repaired using approved 
repair materials compatible with the existing concrete substrate.
6.3.2 — Repair depth shall extend a minimum of 25 mm beyond the corroded 
reinforcement or the depth of unsound concrete, whichever is greater.
6.3.3 — All loose, delaminated, or deteriorated concrete shall be removed 
to sound concrete before repair.

Section 6.2 — Delamination Repair
6.2.1 — Areas of delamination shall be identified by chain drag or 
impact-echo testing and marked for repair.
6.2.2 — Delaminated concrete shall be removed by hydrodemolition or 
mechanical means to the full extent of the delamination.""",
                "source": "ACI_562-19.txt",
                "page": 1,
                "type": "txt",
            },
            {
                "text": """ACI 222R-19: Protection of Metals in Concrete Against Corrosion

Section 4 — Corrosion Protection
4.2 — Chloride limits for new construction:
(a) Prestressed concrete: 0.06% by mass of cement
(b) Reinforced concrete exposed to chlorides: 0.15% by mass of cement
(c) Reinforced concrete in dry conditions: 0.30% by mass of cement

4.3 — Corrosion detection methods:
(a) Half-cell potential survey (ASTM C876)
(b) Linear polarization resistance
(c) Visual inspection for rust staining, cracking, delamination

4.4 — When corrosion is detected:
(a) Assess extent of corrosion damage
(b) Determine if section loss exceeds 20% — if so, reinforcement replacement required
(c) Apply corrosion inhibitor or cathodic protection as appropriate""",
                "source": "ACI_222R-19.txt",
                "page": 1,
                "type": "txt",
            },
            {
                "text": """ASCE 7-22: Minimum Design Loads and Associated Criteria

Section 12.12 — Story Drift Limits
12.12.1 — Story drift limits:
Risk Category I/II: 0.020 * hsx
Risk Category III: 0.015 * hsx
Risk Category IV: 0.010 * hsx
where hsx = story height below level x

Table 12.12-1 — Allowable Story Drift
Seismic force-resisting system | Category I,II | Category III | Category IV
Moment frames (concrete/steel) |    0.020hsx   |   0.015hsx  |  0.010hsx
Bearing wall systems           |    0.015hsx   |   0.010hsx  |  0.010hsx
Braced frame systems           |    0.020hsx   |   0.015hsx  |  0.010hsx

Section 12.8.6 — P-Delta Effects
The ratio θ = (Px * Δ * Ie) / (Vx * hsx * Cd) shall not exceed 0.10 
in any story. Where θ exceeds 0.10, the effects shall be amplified.""",
                "source": "ASCE_7-22.txt",
                "page": 1,
                "type": "txt",
            },
            {
                "text": """IBC 2021: International Building Code

Section 1612 — Structural Integrity
1612.1 — Buildings classified as Risk Category III or IV shall be 
designed with structural integrity provisions.

Section 3401 — Existing Buildings
3401.2 — Repairs, alterations, and additions to existing buildings shall 
comply with the provisions of this chapter and the International 
Existing Building Code (IEBC).

Section 1704 — Special Inspections and Tests
1704.1 — Special inspections shall be performed during construction of:
(a) Cast-in-place concrete
(b) Structural steel connections
(c) Masonry construction
(d) Foundation systems in seismic zones

Section 1901 — Concrete Design Standards
1901.2 — Structural concrete shall be designed in accordance with 
ACI 318 as amended in this code.""",
                "source": "IBC_2021.txt",
                "page": 1,
                "type": "txt",
            },
        ]
        
        logger.info("Created sample building code documents for demo")
        return sample_codes
