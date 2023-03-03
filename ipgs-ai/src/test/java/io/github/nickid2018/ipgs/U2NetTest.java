package io.github.nickid2018.ipgs;

import io.github.nickid2018.ipgs.dataset.TrainDataSetIterator;
import io.github.nickid2018.ipgs.network.U2Net;
import lombok.SneakyThrows;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.split.StreamInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class U2NetTest {

    public static final File MODEL_FILE = new File("model.zip");
    public static final File TRAIN_FILE = new File("train-mini-mini-mini.zip");
    public static final Logger LOGGER = LoggerFactory.getLogger(U2NetTest.class);

    @Test
    @SneakyThrows
    public void test() {
        System.setProperty("org.bytedeco.javacpp.maxPhysicalBytes", "8G");
        ComputationGraph graph;
        if (MODEL_FILE.exists()) {
            LOGGER.info("Found model file, loading...");
            graph = ComputationGraph.load(MODEL_FILE, true);
        } else {
            LOGGER.info("Model file not found, training...");

            graph = U2Net.initNETP(256, 256, 3);

            StatsStorage statsStorage = new InMemoryStatsStorage();

            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(statsStorage);

            TrainDataSetIterator iterator = new TrainDataSetIterator(TRAIN_FILE, 1, 5711);
            graph.setListeners(new StatsListener(statsStorage));
            graph.fit(iterator, 100);
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
