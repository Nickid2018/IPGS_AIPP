package io.github.nickid2018.ipgs;

import io.github.nickid2018.ipgs.dataset.TrainDataSetIterator;
import io.github.nickid2018.ipgs.network.U2Net;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

@Slf4j
public class U2NetTest {

    public static final File MODEL_FILE = new File("model.zip");
    public static final File TRAIN_FILE = new File("train.zip");

    @Test
    @SneakyThrows
    public void test() {
        System.setProperty("org.bytedeco.javacpp.maxBytes", "8G");
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        ComputationGraph graph;
        if (MODEL_FILE.exists()) {
            log.info("Found model file, loading...");
            graph = ComputationGraph.load(MODEL_FILE, true);
        } else {
            log.info("Model file not found, training...");

            graph = U2Net.initNETP(256, 256, 3);

            StatsStorage statsStorage = null;

            try {
                UIServer uiServer = UIServer.getInstance();
                statsStorage = new InMemoryStatsStorage();
                uiServer.attach(statsStorage);
            } catch (Exception e) {
                log.error("Failed to start UI server, maybe on Colab?");
            }

            TrainDataSetIterator iterator = new TrainDataSetIterator(TRAIN_FILE, 1, 5386);
            if (statsStorage != null)
                graph.setListeners(new StatsListener(statsStorage));
            graph.fit(iterator, 1);

            log.info("Training finished, saving model...");
            graph.save(MODEL_FILE, true);
        }

//        ComputationGraph graph = UNet.builder().inputShape(new int[]{3, 256, 256}).build().init();
//
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//
//        TrainDataSetIterator iterator = new TrainDataSetIterator(TRAIN_FILE, 1, 5386);
//        graph.setListeners(new StatsListener(statsStorage));
//        graph.fit(iterator, 1);
//        graph.save(MODEL_FILE, true);
    }
}
