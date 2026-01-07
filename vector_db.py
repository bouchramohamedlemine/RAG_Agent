from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url="http://127.0.0.1:6333", collection="docs", dim=3072):
        """
        Turn text docs into vectors and compare them against each other using Distance.COSINE.
        Vectors closer to each other in the vector space should have some similaririty,
        so we pull their original text data and pass it to the LLM.
        
        """

        # In production change the URL
        self.client = QdrantClient(url=url, timeout=30)
        # if connection is not established within 30 secs the program crashes

        self.collection = collection
        self.dim = dim

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )




    def clear(self):
        """
            Clear all points in the collection by dropping and recreating it.
        """
        import time
        
        # Delete the collection if it exists
        if self.client.collection_exists(self.collection):
            try:
                self.client.delete_collection(self.collection)
                # Wait a moment for the deletion to complete
                time.sleep(0.2)
            except Exception as e:
                # If deletion fails, try to continue anyway
                pass

        # Only create if it doesn't exist (double-check after deletion)
        if not self.client.collection_exists(self.collection):
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
            except Exception as e:
                # If collection already exists (race condition), that's fine
                # The collection is already cleared/empty, so we can continue
                if "already exists" not in str(e).lower():
                    raise

    def upsert(self, ids, vectors, payloads):
        """
            Insert and update
        """
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)



    def search(self, query_vector, top_k: int = 5):
        """
            Search 5 results from the vector DB
        """
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()

        # QueryResponse has a 'points' attribute containing the results
        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}