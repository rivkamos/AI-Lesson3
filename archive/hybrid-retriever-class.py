# from rank_bm25 import BM25Okapi
 
# class HybridRetriever:
#     def __init__(self, collection, documents):
#         self.collection = collection
#         self.bm25 = BM25Okapi([doc.split() for doc in documents])
        
#     def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
#         # Vector search
#         query_emb = create_query_embedding(query)
#         vector_results = self.collection.query(
#             query_embeddings=[query_emb],
#             n_results=k*2
#         )
        
#         # BM25 search
#         bm25_scores = self.bm25.get_scores(query.split())
        
#         # Reciprocal rank fusion
#         fused_scores = self.reciprocal_rank_fusion(vector_results, bm25_scores, alpha)
#         return fused_scores[:k]