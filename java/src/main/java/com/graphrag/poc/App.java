package com.graphrag.poc;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.List;

import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Record;
import org.neo4j.driver.Result;
import org.neo4j.driver.Session;

import com.graphrag.poc.adapter.MilvusVectorStoreAdapter;
import com.graphrag.poc.embedding.SimpleEmbeddingModel;
import com.graphrag.poc.store.MilvusEmbeddingStore;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.retriever.Retriever;

public class App {
    public static void main(String[] args) {
        System.out.println("GraphRAG Java PoC starting...");

        String uri = "bolt://localhost:7687";
        String user = "neo4j";
        String password = "test";

        try (Driver driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password))) {
            try (Session session = driver.session()) {
                Result result = session.run("RETURN 'hello' AS msg");
                Record record = result.single();
                System.out.println("Neo4j test query result: " + record.get("msg").asString());
            }
        } catch (Exception e) {
            System.err.println("Failed to connect to Neo4j: " + e.getMessage());
        }

        String milvusHost = "localhost";
        int milvusPort = 19530;
        boolean milvusReachable = false;
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(milvusHost, milvusPort), 3000);
            milvusReachable = true;
            System.out.println("Milvus reachable at " + milvusHost + ":" + milvusPort);
        } catch (IOException e) {
            System.err.println("Milvus not reachable at " + milvusHost + ":" + milvusPort + " - " + e.getMessage());
        }

        if (milvusReachable) {
            try {
                MilvusVectorStoreAdapter adapter = new MilvusVectorStoreAdapter(milvusHost, milvusPort, "graphrag_java_poc", 3);
                adapter.initialize();

                SimpleEmbeddingModel embeddingModel = new SimpleEmbeddingModel();
                MilvusEmbeddingStore embeddingStore = new MilvusEmbeddingStore(adapter);

                TextSegment document = TextSegment.from("Milvus is a Java-friendly vector database for retrieval pipelines");
                Embedding documentEmbedding = embeddingModel.embed(document.text()).content();
                embeddingStore.add(documentEmbedding, document);

                Retriever<TextSegment> retriever = EmbeddingStoreRetriever.from(embeddingStore, embeddingModel, 3);
                List<TextSegment> relevant = retriever.findRelevant("Java-friendly vector database");

                System.out.println("Retrieval results from langchain4j:");
                for (TextSegment segment : relevant) {
                    System.out.println("- " + segment.text());
                }
            } catch (Exception e) {
                System.err.println("Milvus adapter failed: " + e.getMessage());
            }
        }

        System.out.println("PoC complete. Next: replace the stub embedding model with a hosted embedding provider.");
    }
}
