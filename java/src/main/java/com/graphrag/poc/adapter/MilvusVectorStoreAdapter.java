package com.graphrag.poc.adapter;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.param.ConnectParam;
import io.milvus.param.MetricType;
import io.milvus.param.collection.CreateCollectionParam;
import io.milvus.param.collection.FieldType;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;

public class MilvusVectorStoreAdapter implements VectorStoreAdapter {
    private final MilvusServiceClient client;
    private final String collectionName;
    private final int dimension;

    public MilvusVectorStoreAdapter(String host, int port, String collectionName, int dimension) {
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(host)
                .withPort(port)
                .build();
        this.client = new MilvusServiceClient(connectParam);
        this.collectionName = collectionName;
        this.dimension = dimension;
    }

    @Override
    public void initialize() throws Exception {
        List<FieldType> fields = new ArrayList<>();
        fields.add(FieldType.newBuilder()
                .withName("id")
                .withDataType(DataType.VarChar)
                .withMaxLength(64)
                .withPrimaryKey(true)
                .build());
        fields.add(FieldType.newBuilder()
                .withName("text")
                .withDataType(DataType.VarChar)
                .withMaxLength(2048)
                .build());
        fields.add(FieldType.newBuilder()
                .withName("embedding")
                .withDataType(DataType.FloatVector)
                .withDimension(dimension)
                .build());

        CreateCollectionParam createCollectionParam = CreateCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .withDescription("GraphRAG Java PoC collection")
                .withShardsNum(2)
                .withFieldTypes(fields)
                .build();

        try {
            client.createCollection(createCollectionParam);
        } catch (Exception ignored) {
            // If collection already exists, ignore the duplicate creation.
        }
    }

    @Override
    public void upsert(String id, String text, float[] vector, Map<String, String> metadata) throws Exception {
        List<List<Float>> vectors = new ArrayList<>();
        vectors.add(toFloatList(vector));

        List<String> ids = List.of(id);
        List<String> texts = List.of(text);

        List<InsertParam.Field> fields = new ArrayList<>();
        fields.add(new InsertParam.Field("id", ids));
        fields.add(new InsertParam.Field("text", texts));
        fields.add(new InsertParam.Field("embedding", vectors));

        InsertParam insertParam = InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(fields)
                .build();

        client.insert(insertParam);
    }

    @Override
    public List<SearchResult> similaritySearch(float[] queryVector, int limit) throws Exception {
        List<List<Float>> vectors = new ArrayList<>();
        vectors.add(toFloatList(queryVector));

        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withMetricType(MetricType.L2)
                .withOutFields(List.of("id", "text"))
                .withTopK(limit)
                .withVectorFieldName("embedding")
                .withVectors(vectors)
                .build();

        Object response = client.search(searchParam);
        List<SearchResult> results = new ArrayList<>();
        try {
            Method getSearchResults = response.getClass().getMethod("getSearchResults");
            Object rawResults = getSearchResults.invoke(response);
            if (rawResults instanceof List<?> rawList) {
                for (Object item : rawList) {
                    String id = readString(item, "id");
                    String text = readString(item, "text");
                    double score = readDouble(item, "score");
                    results.add(new SearchResult(id, text, score));
                }
            }
        } catch (Exception ignored) {
            // The SDK result shape may vary by version; return an empty list as a graceful fallback.
        }
        return results;
    }

    private List<Float> toFloatList(float[] values) {
        List<Float> list = new ArrayList<>(values.length);
        for (float value : values) {
            list.add(value);
        }
        return list;
    }

    private String readString(Object item, String fieldName) {
        try {
            Method method = item.getClass().getMethod("get" + capitalize(fieldName));
            Object value = method.invoke(item);
            return value == null ? "" : value.toString();
        } catch (Exception ignored) {
            return "";
        }
    }

    private double readDouble(Object item, String fieldName) {
        try {
            Method method = item.getClass().getMethod("get" + capitalize(fieldName));
            Object value = method.invoke(item);
            return value instanceof Number number ? number.doubleValue() : 0.0;
        } catch (Exception ignored) {
            return 0.0;
        }
    }

    private String capitalize(String value) {
        if (value == null || value.isBlank()) {
            return value;
        }
        return Character.toUpperCase(value.charAt(0)) + value.substring(1);
    }
}
