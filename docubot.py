"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

class DocuBot:
    # Common English stop words to ignore during tokenization
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the',
        'to', 'was', 'will', 'with', 'you', 'i', 'me', 'my', 'we', 'us', 'this'
    }

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)
        
        # Split documents into sections for finer-grained retrieval
        self.sections = self._build_sections()  # List of (filename, section_text, section_id)

        # Build a retrieval index at the section level
        self.index = self.build_index(self.sections)
        
        # Confidence threshold: minimum score to return a result
        # Too low = confident but wrong answers; too high = "I don't know" too often
        self.min_score = 2

    def _build_sections(self):
        """
        Convert all documents into sections for retrieval.
        """
        all_sections = []
        for filename, text in self.documents:
            sections = self._split_into_sections(text, filename)
            all_sections.extend(sections)
        return all_sections

    def _tokenize(self, text):
        """
        Convert text to lowercase tokens, removing simple punctuation.
        Filter out common stop words.
        """
        normalized = text.lower()
        for ch in ".,;:!?()[]{}<>\"'\\/":
            normalized = normalized.replace(ch, " ")
        tokens = [token for token in normalized.split() if token and token not in self.STOP_WORDS]
        return tokens

    def _split_into_sections(self, text, filename):
        """
        Split a document into sections using markdown heading boundaries (##).
        Each section includes its heading and the text until the next heading.
        
        Returns a list of (filename, section_text, section_id) tuples.
        """
        sections = []
        lines = text.split('\n')
        current_section = []
        section_count = 0
        
        for line in lines:
            # Check if this line is a level 2+ markdown heading
            if line.startswith('##'):
                # Save previous section if it has content
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        sections.append((filename, section_text, f"{filename}#{section_count}"))
                        section_count += 1
                current_section = [line]
            else:
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append((filename, section_text, f"{filename}#{section_count}"))
        
        # If no sections were found (no headers), treat entire text as one section
        if not sections:
            sections.append((filename, text.strip(), f"{filename}#0"))
        
        return sections

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, sections):
        """
        Build an inverted index at the section level, mapping lowercase words 
        to the section identifiers (section_id) that contain them.

        sections is a list of (filename, section_text, section_id) tuples.
        """
        index = {}
        for filename, section_text, section_id in sections:
            tokens = self._tokenize(section_text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                index.setdefault(token, []).append(section_id)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0

        text_tokens = self._tokenize(text)
        score = 0
        for token in query_tokens:
            score += text_tokens.count(token)
        return score

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant section snippets.
        
        Only returns sections with score >= min_score (confidence guardrail).
        Returns a list of (filename, section_text) sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Find candidate section IDs from the index
        candidate_section_ids = set()
        for token in query_tokens:
            candidate_section_ids.update(self.index.get(token, []))

        # Score each candidate section
        scored = []
        for filename, section_text, section_id in self.sections:
            if section_id not in candidate_section_ids:
                continue
            score = self.score_document(query, section_text)
            if score >= self.min_score:  # Confidence guardrail
                scored.append((score, filename, section_text))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [(filename, text) for _, filename, text in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
