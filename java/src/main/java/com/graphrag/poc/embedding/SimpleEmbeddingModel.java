package com.graphrag.poc.embedding;

import java.util.ArrayList;
import java.util.List;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;

public class SimpleEmbeddingModel implements EmbeddingModel {
    @Override
    public Response<Embedding> embed(String text) {
        return Response.from(Embedding.from(toVector(text)));
    }

    @Override
    public Response<Embedding> embed(TextSegment textSegment) {
        return embed(textSegment.text());
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        List<Embedding> embeddings = new ArrayList<>(textSegments.size());
        for (TextSegment textSegment : textSegments) {
            embeddings.add(embed(textSegment.text()).content());
        }
        return Response.from(embeddings);
    }

    @Override
    public int dimension() {
        return 3;
    }

    private float[] toVector(String text) {
        int hash = text.toLowerCase().hashCode();
        float v1 = (float) ((((hash % 97) / 97.0) * 2.0) - 1.0);
        float v2 = (float) (((hash * 31) % 89) / 89.0);
        float v3 = (float) (((hash * 17) % 53) / 53.0);
        return new float[]{v1, v2, v3};
    }
}
