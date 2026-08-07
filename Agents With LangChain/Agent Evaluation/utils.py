from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def get_sentence_similarity(sentence1: str, sentence2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Encodes two sentences into embeddings using a SentenceTransformer model
    and returns their cosine similarity.
    """
    model = SentenceTransformer(model_name)

    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)

    similarity = cos_sim(embeddings[0], embeddings[1])

    return similarity.item()


if __name__ == "__main__":
    s1 = "What is the inventory status of iPhone 15?"
    s2 = "Do you have iPhone 15 in stock?"

    score = get_sentence_similarity(s1, s2)
    print(f"Cosine similarity: {score:.4f}")