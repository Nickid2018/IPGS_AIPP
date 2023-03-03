package io.github.nickid2018.ipgs.dataset;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.File;
import java.io.IOException;

public class TrainDataSetIterator extends BaseDatasetIterator {

    public TrainDataSetIterator(File file, int batchSize, int numExamples) throws IOException {
        super(batchSize, numExamples, new TrainDataFetcher(file, numExamples));
    }

    public TrainDataFetcher getFetcher() {
        return (TrainDataFetcher) fetcher;
    }
}
