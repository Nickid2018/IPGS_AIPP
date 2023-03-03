package io.github.nickid2018.ipgs;

import io.github.nickid2018.ipgs.dataset.TrainDataSetIterator;
import io.github.nickid2018.ipgs.network.U2Net;
import lombok.SneakyThrows;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.api.preprocessor.ImageMultiPreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class U2NetTest {

    public static final File MODEL_FILE = new File("model.zip");
    public static final File TRAIN_FILE = new File("train-mini-mini.zip");
    public static final Logger LOGGER = LoggerFactory.getLogger(U2NetTest.class);

    @Test
    @SneakyThrows
    public void test() {
        System.setProperty("org.bytedeco.javacpp.maxPhysicalBytes", "5G");
        ComputationGraph graph;
        if (MODEL_FILE.exists()) {
            LOGGER.info("Found model file, loading...");
            graph = ComputationGraph.load(MODEL_FILE, true);
        } else {
            LOGGER.info("Model file not found, training...");
            graph = U2Net.init(512, 512, 3);

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);

            TrainDataSetIterator iterator = new TrainDataSetIterator(TRAIN_FILE, 1, 5386);
            graph.setListeners(new StatsListener(statsStorage));
            graph.fit(iterator, 1);
            graph.save(MODEL_FILE, true);
        }
    }
}
