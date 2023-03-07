package io.github.nickid2018.ipgs;

import io.github.nickid2018.ipgs.network.U2Net;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

@Slf4j
public class U2NetOverTest {

    public static File MODEL_FILE = new File("model.zip");

    private INDArray loadImage(String path) throws IOException {
        BufferedImage image = ImageIO.read(new File(path));
        float[][][][] imageArray = new float[1][3][image.getHeight()][image.getWidth()];
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                int rgb = image.getRGB(i, j);
                imageArray[0][0][j][i] = (rgb >> 16) & 0xFF;
                imageArray[0][1][j][i] = (rgb >> 8) & 0xFF;
                imageArray[0][2][j][i] = rgb & 0xFF;
            }
        }
        return Nd4j.create(imageArray).divi(255);
    }

    @Test
    @SneakyThrows
    public void test() {
        System.setProperty("org.bytedeco.javacpp.maxBytes", "8G");

        ComputationGraph graph;
        if (MODEL_FILE.exists()) {
            log.info("Found model file, loading...");
            graph = ComputationGraph.load(MODEL_FILE, true);
        } else {
            log.error("Model file not found");
            return;
        }
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        INDArray[] result = graph.output(loadImage("test.png"));
        INDArray array = result[0];
        BufferedImage image = new BufferedImage(256, 256, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < array.shape()[2]; i++) {
            for (int j = 0; j < array.shape()[3]; j++) {
                float value = array.getFloat(0, 0, i, j);
                int rgb = (int) (value * 255);
                image.setRGB(j, i, new Color(rgb, rgb, rgb).getRGB());
            }
        }
        ImageIO.write(image, "png", new File("testOver.png"));
    }
}
