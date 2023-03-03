package io.github.nickid2018.ipgs.dataset;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipFile;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

@Slf4j
public class TrainDataFetcher extends BaseDataFetcher {

    public static final int IMAGE_SIZE = 256;

    private float[][][][] images = new float[1][3][IMAGE_SIZE][IMAGE_SIZE];
    private float[][][][] labels = new float[1][1][IMAGE_SIZE][IMAGE_SIZE];

    @Getter
    private int predictBuffer = 1;

    private final ZipFile zipFile;

    public TrainDataFetcher(File file, int totalExamples) throws IOException {
        zipFile = new ZipFile(file);
        this.totalExamples = totalExamples;
        numOutcomes = IMAGE_SIZE * IMAGE_SIZE;
        inputColumns = IMAGE_SIZE * IMAGE_SIZE * 3;
    }

    public void setPredictBuffer(int predictBuffer) {
        if (this.predictBuffer == predictBuffer)
            return;
        Preconditions.checkArgument(predictBuffer > 0, "Predict buffer must be positive");
        this.predictBuffer = predictBuffer;
        images = new float[predictBuffer][3][IMAGE_SIZE][IMAGE_SIZE];
        labels = new float[predictBuffer][1][IMAGE_SIZE][IMAGE_SIZE];
    }

    @Override
    public void fetch(int numExamples) {
        System.out.println("\33[0;33mNow cursor is " + cursor + "\33[0m");
        System.out.flush();

        int actualExamples = Math.min(numExamples, totalExamples - cursor);

        if (predictBuffer < actualExamples) {
            images = new float[actualExamples][3][IMAGE_SIZE][IMAGE_SIZE];
            labels = new float[actualExamples][1][IMAGE_SIZE][IMAGE_SIZE];
            predictBuffer = actualExamples;
        }

        for (int i = 0; i < actualExamples; i++, cursor++) {
            try {
                readImage(cursor, i);
                readLabel(cursor, i);
            } catch (IOException e) {
                log.error("Error while reading image", e);
                images[i] = new float[3][IMAGE_SIZE][IMAGE_SIZE];
                labels[i] = new float[1][IMAGE_SIZE][IMAGE_SIZE];
            }
        }

        INDArray features;
        INDArray labelsArray;

        if (images.length == actualExamples) {
            features = Nd4j.create(images);
            labelsArray = Nd4j.create(labels);
        } else {
            features = Nd4j.create(Arrays.copyOfRange(images, 0, actualExamples));
            labelsArray = Nd4j.create(Arrays.copyOfRange(labels, 0, actualExamples));
        }

        features = features.divi(255);
        labelsArray = labelsArray.gt(30).castTo(DataType.FLOAT);

        curr = new DataSet(features, labelsArray);
    }

    private void readImage(int index, int bufferIndex) throws IOException {
        ZipArchiveEntry entry = zipFile.getEntry("image/" + index + ".png");
        BufferedImage image = ImageIO.read(zipFile.getInputStream(entry));
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                int rgb = image.getRGB(i, j);
                images[bufferIndex][0][j][i] = (byte) ((rgb >> 16) & 0xFF);
                images[bufferIndex][1][j][i] = (byte) ((rgb >> 8) & 0xFF);
                images[bufferIndex][2][j][i] = (byte) (rgb & 0xFF);
            }
        }
    }

    private void readLabel(int index, int bufferIndex) throws IOException {
        ZipArchiveEntry entry = zipFile.getEntry("mask/" + index + ".png");
        BufferedImage image = ImageIO.read(zipFile.getInputStream(entry));
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                int rgb = image.getRGB(i, j);
                labels[bufferIndex][0][j][i] = (byte) ((rgb >> 16) & 0xFF);
            }
        }
    }
}
