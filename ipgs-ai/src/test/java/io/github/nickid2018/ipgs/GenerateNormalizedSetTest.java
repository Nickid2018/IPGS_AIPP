package io.github.nickid2018.ipgs;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarFile;
import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;

public class GenerateNormalizedSetTest {

    public BufferedImage toBitmap(String data) throws DataFormatException, IOException {
        byte[] bytes = Base64.getDecoder().decode(data);

        ByteArrayOutputStream o = new ByteArrayOutputStream(bytes.length);
        Inflater inflater = new Inflater();
        inflater.reset();
        inflater.setInput(bytes);
        byte[] buf = new byte[1024];
        while (!inflater.finished()) {
            int i = inflater.inflate(buf);
            o.write(buf, 0, i);
        }
        inflater.end();

        return ImageIO.read(new ByteArrayInputStream(o.toByteArray()));
    }

    @Test
    public void scale() throws IOException {
        String from = "train-mini.zip";
        String to = "train-mini-mini.zip";
        ZipFile zipFile = new ZipFile(from);
        ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(to));
        int size = (int) zipFile.stream().count();
        int counter = 0;
        for (ZipEntry entry : zipFile.stream().toList()) {
            counter++;
            System.out.println("\33[0;33m Progress: " + counter + "/" + size + " \33[0m");
            System.out.flush();
            if (entry.isDirectory())
                continue;
            BufferedImage image = ImageIO.read(zipFile.getInputStream(entry));
            BufferedImage scaled = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = scaled.createGraphics();
            g.drawImage(image, 0, 0, 512, 512, null);
            zipOutputStream.putNextEntry(new ZipEntry(entry.getName()));
            ImageIO.write(scaled, "png", zipOutputStream);
            zipOutputStream.closeEntry();
        }
        zipOutputStream.close();
    }

    @Test
    public void tar() throws IOException {
        String from = "Supervisely Person Dataset.tar";
        String to = "train.zip";

        ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(to));
        int progress = 0;
        try (TarFile tarFile = new TarFile(new File(from))) {
            Set<TarArchiveEntry> entries = new HashSet<>();
            Map<String, TarArchiveEntry> others = new HashMap<>();
            tarFile.getEntries().forEach(entry -> {
                if (entry.getName().endsWith(".json") && !entry.getName().endsWith("meta.json"))
                    entries.add(entry);
                else
                    others.put(entry.getName(), entry);
            });
            ENTRY: for (TarArchiveEntry entry : entries) {
                String name = entry.getName();
                String imagePath = name.replace("/ann/", "/img/").substring(0, name.length() - 5);
                TarArchiveEntry image = others.get(imagePath);

                // Filter > 4096
                JsonObject object = new JsonParser().parse(IOUtils.toString(
                        tarFile.getInputStream(entry), StandardCharsets.UTF_8)).getAsJsonObject();
                JsonObject size = object.get("size").getAsJsonObject();
                if (size.get("width").getAsInt() > 4096 || size.get("height").getAsInt() > 4096) {
                    progress++;
                    continue;
                }

                boolean needScale = size.get("width").getAsInt() > 2048 || size.get("height").getAsInt() > 2048;

                BufferedImage image1 = ImageIO.read(tarFile.getInputStream(image));
                if (image1 == null) {
                    System.out.println("Image can't be read: " + imagePath);
                    progress++;
                    continue;
                }
                BufferedImage deal = new BufferedImage(2048, 2048, BufferedImage.TYPE_INT_RGB);
                if (needScale)
                    deal.getGraphics().drawImage(image1.getScaledInstance(
                                    image1.getWidth() / 2, image1.getHeight() / 2, BufferedImage.SCALE_AREA_AVERAGING),
                            0, 0, null);
                else
                    deal.getGraphics().drawImage(image1, 0, 0, null);

                BufferedImage type = new BufferedImage(needScale ? 4096 : 2048, needScale ? 4096 : 2048, BufferedImage.TYPE_BYTE_BINARY);
                JsonArray objects = object.get("objects").getAsJsonArray();
                for (JsonElement element : objects) {
                    String geometryType = element.getAsJsonObject().get("geometryType").getAsString();
                    if (geometryType.equals("bitmap")) {
                        String data = element.getAsJsonObject().get("bitmap").getAsJsonObject().get("data").getAsString();
                        try {
                            BufferedImage bitmap = toBitmap(data);
                            JsonArray origin = element.getAsJsonObject().get("bitmap").getAsJsonObject().get("origin").getAsJsonArray();
                            int x = origin.get(0).getAsInt();
                            int y = origin.get(1).getAsInt();
                            type.getGraphics().drawImage(bitmap, x, y, null);
                        } catch (DataFormatException e) {
                            System.err.println("Data format error: " + imagePath);
                            e.printStackTrace();
                            progress++;
                            continue ENTRY;
                        }
                    } else if (geometryType.equals("polygon")) {
                        JsonArray points = element.getAsJsonObject().get("points").getAsJsonObject().get("exterior").getAsJsonArray();
                        int[] x = new int[points.size()];
                        int[] y = new int[points.size()];
                        for (int i = 0; i < points.size(); i++) {
                            x[i] = points.get(i).getAsJsonArray().get(0).getAsInt();
                            y[i] = points.get(i).getAsJsonArray().get(1).getAsInt();
                        }
                        Graphics2D graphics = (Graphics2D) type.getGraphics();
                        graphics.setColor(Color.WHITE);
                        graphics.fillPolygon(x, y, points.size());
                        graphics.setColor(Color.BLACK);
                        JsonArray holes = element.getAsJsonObject().get("points").getAsJsonObject().get("interior").getAsJsonArray();
                        for (JsonElement hole : holes) {
                            JsonArray holePoints = hole.getAsJsonArray();
                            int[] hx = new int[holePoints.size()];
                            int[] hy = new int[holePoints.size()];
                            for (int i = 0; i < holePoints.size(); i++) {
                                hx[i] = holePoints.get(i).getAsJsonArray().get(0).getAsInt();
                                hy[i] = holePoints.get(i).getAsJsonArray().get(1).getAsInt();
                            }
                            graphics.fillPolygon(hx, hy, holePoints.size());
                        }
                    } else {
                        System.err.println("Unsupported geometry type: " + geometryType);
                        progress++;
                        continue ENTRY;
                    }
                }

                if (needScale) {
                    Image scaledInstance = type.getScaledInstance(2048, 2048, BufferedImage.SCALE_AREA_AVERAGING);
                    type = new BufferedImage(2048, 2048, BufferedImage.TYPE_BYTE_BINARY);
                    type.getGraphics().drawImage(scaledInstance, 0, 0, null);
                }

                ZipEntry saveEntry = new ZipEntry("image/" + progress + ".png");
                zipOutputStream.putNextEntry(saveEntry);
                ImageIO.write(deal, "png", zipOutputStream);
                zipOutputStream.closeEntry();

                saveEntry = new ZipEntry("mask/" + progress + ".png");
                zipOutputStream.putNextEntry(saveEntry);
                ImageIO.write(type, "png", zipOutputStream);
                zipOutputStream.closeEntry();

                progress++;
                System.out.print("\33[1A");
                System.out.flush();
                System.out.println("\33[0;33m Progress: " + progress + "/" + entries.size() + " \33[0m");
                System.out.flush();
            }
        }
        zipOutputStream.close();
    }
}
