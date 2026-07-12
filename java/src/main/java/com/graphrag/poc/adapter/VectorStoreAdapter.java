package com.graphrag.poc.adapter;

import java.util.List;
import java.util.Map;

public interface VectorStoreAdapter {
    void initialize() throws Exception;

    void upsert(String id, String text, float[] vector, Map<String, String> metadata) throws Exception;

    List<SearchResult> similaritySearch(float[] queryVector, int limit) throws Exception;
}
