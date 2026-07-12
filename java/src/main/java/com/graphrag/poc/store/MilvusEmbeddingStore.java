package com.graphrag.poc.store;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import com.graphrag.poc.adapter.MilvusVectorStoreAdapter;
import com.graphrag.poc.adapter.SearchResult;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;

public class MilvusEmbeddingStore implements EmbeddingStore<TextSegment> {

    private final MilvusVectorStoreAdapter adapter;
    private final Map<String, TextSegment> segmentsById = new HashMap<>();

    public MilvusEmbeddingStore(MilvusVectorStoreAdapter adapter) {
        this.adapter = adapter;
    }

    @Override
    public String add(Embedding embedding) {
        String id = nextId();
        try {
            adapter.upsert(id, "", embedding.vector(), Map.of());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        segmentsById.put(id, TextSegment.from(""));
        return id;
    }

    @Override
    public void add(String id, Embedding embedding) {
        try {
            adapter.upsert(id, "", embedding.vector(), Map.of());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        segmentsById.put(id, TextSegment.from(""));
    }

    @Override
    public String add(Embedding embedding, TextSegment embedded) {
        String id = nextId();
        try {
            adapter.upsert(id, embedded.text(), embedding.vector(), Map.of());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        segmentsById.put(id, embedded);
        return id;
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings) {
        List<String> ids = new ArrayList<>();
        for (Embedding embedding : embeddings) {
            ids.add(add(embedding));
        }
        return ids;
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings, List<TextSegment> embedded) {
        if (embeddings.size() != embedded.size()) {
            throw new IllegalArgumentException("Embeddings and documents must have the same length");
        }
        List<String> ids = new ArrayList<>();
        for (int i = 0; i < embeddings.size(); i++) {
            ids.add(add(embeddings.get(i), embedded.get(i)));
        }
        return ids;
    }

    @Override
    public EmbeddingSearchResult<TextSegment> search(EmbeddingSearchRequest request) {
        List<SearchResult> milvusResults;
        try {
            milvusResults = adapter.similaritySearch(request.queryEmbedding().vector(), request.maxResults());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        List<EmbeddingMatch<TextSegment>> matches = new ArrayList<>();

        for (SearchResult result : milvusResults) {
            TextSegment segment = segmentsById.getOrDefault(result.id(), TextSegment.from(result.text()));
            matches.add(new EmbeddingMatch<>(result.score(), result.id(), request.queryEmbedding(), segment));
        }

        return new EmbeddingSearchResult<>(matches);
    }

    private String nextId() {
        return UUID.randomUUID().toString();
    }
}
