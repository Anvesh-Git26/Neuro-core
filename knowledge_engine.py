import faiss
from sentence_transformers import SentenceTransformer

class KnowledgeEngine:
    def __init__(self):
        print("Loading Knowledge Base...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.facts = [
            "The sun is hot because of nuclear fusion.",
            "A spider has eight legs and uses webs to catch prey.",
            "The telephone was invented by Alexander Graham Bell.",
            "The blue whale is the largest animal on Earth.",
            "Water freezes at 0 degrees Celsius.",
            "Miko is a companion robot created to help children learn."
        ]
        self._build_index()

    def _build_index(self):
        fact_vectors = self.embedder.encode(self.facts)
        d = fact_vectors.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(fact_vectors)

    def search(self, query):
        query_vector = self.embedder.encode([query])
        D, I = self.index.search(query_vector, k=1)
        return self.facts[I[0][0]]

if __name__ == "__main__":
    engine = KnowledgeEngine()
    print(engine.search("Why is the sun burning?"))