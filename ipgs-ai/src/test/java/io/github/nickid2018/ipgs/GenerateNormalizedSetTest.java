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
    public void tar() throws IOException {
        String from = "Supervisely Person Dataset.tar";
        String to = "train.zip";

        ZipOutputStream zipOutputStream = new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(to)));
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

                int width = size.get("width").getAsInt();
                int height = size.get("height").getAsInt();
                if (width > 4096 || height > 4096) {
                    progress++;
                    continue;
                }

                int scale = 1;
                if (width > 2048 || height > 2048) {
                    scale = 16;
                } else if (width > 1024 || height > 1024) {
                    scale = 8;
                } else if (width > 512 || height > 512) {
                    scale = 4;
                } else if (width > 256 || height > 256) {
                    scale = 2;
                }

                BufferedImage image1 = ImageIO.read(tarFile.getInputStream(image));
                if (image1 == null) {
                    System.out.println("Image can't be read: " + imagePath);
                    progress++;
                    continue;
                }

                BufferedImage deal = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
                deal.getGraphics().drawImage(image1, 0, 0, width / scale, height / scale, null);

                BufferedImage type = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);
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

                BufferedImage ctype = new BufferedImage(256, 256, BufferedImage.TYPE_BYTE_BINARY);
                ctype.getGraphics().drawImage(type, 0, 0, width / scale, height / scale, null);

                ZipEntry saveEntry = new ZipEntry("image/" + progress + ".png");
                zipOutputStream.putNextEntry(saveEntry);
                ImageIO.write(deal, "png", zipOutputStream);
                zipOutputStream.closeEntry();

                saveEntry = new ZipEntry("mask/" + progress + ".png");
                zipOutputStream.putNextEntry(saveEntry);
                ImageIO.write(ctype, "png", zipOutputStream);
                zipOutputStream.closeEntry();

                progress++;
                System.out.println("\33[0;33m Progress: " + progress + "/" + entries.size() + " \33[0m");
                System.out.flush();
            }
        }
        zipOutputStream.close();
    }
}
