package io.github.nickid2018.ipgs.network;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class U2Net {

    public static ComputationGraph init(int width, int height, int depth) {
        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADADELTA)
                .weightInit(WeightInit.RELU)
                .l2(5e-5)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .graphBuilder();

        graph.addInputs("input").setInputTypes(InputType.convolutional(height, width, depth));

        RSU7(graph, "stage1", "input", 3, 32, 64);
        graph.addLayer("stage1_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage1");
        RSU6(graph, "stage2", "stage1_pool", 64, 32, 128);
        graph.addLayer("stage2_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage2");
        RSU5(graph, "stage3", "stage2_pool", 128, 64, 256);
        graph.addLayer("stage3_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage3");
        RSU4(graph, "stage4", "stage3_pool", 256, 128, 512);
        graph.addLayer("stage4_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage4");
        RSU4F(graph, "stage5", "stage4_pool", 512, 256, 512);
        graph.addLayer("stage5_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage5");
        RSU4F(graph, "stage6", "stage5_pool", 512, 256, 512);
        // Decoder ---------------------------
        graph.addLayer("stage6d_up", new Upsampling2D.Builder(2).build(), "stage6");
        graph.addVertex("stage5_merge", new MergeVertex(), "stage6d_up", "stage5");
        RSU4F(graph, "stage5d", "stage5_merge", 1024, 256, 512);
        graph.addLayer("stage5d_up", new Upsampling2D.Builder(2).build(), "stage5d");
        graph.addVertex("stage4_merge", new MergeVertex(), "stage5d_up", "stage4");
        RSU4(graph, "stage4d", "stage4_merge", 1024, 128, 256);
        graph.addLayer("stage4d_up", new Upsampling2D.Builder(2).build(), "stage4d");
        graph.addVertex("stage3_merge", new MergeVertex(), "stage4d_up", "stage3");
        RSU5(graph, "stage3d", "stage3_merge", 512, 64, 128);
        graph.addLayer("stage3d_up", new Upsampling2D.Builder(2).build(), "stage3d");
        graph.addVertex("stage2_merge", new MergeVertex(), "stage3d_up", "stage2");
        RSU6(graph, "stage2d", "stage2_merge", 256, 32, 64);
        graph.addLayer("stage2d_up", new Upsampling2D.Builder(2).build(), "stage2d");
        graph.addVertex("stage1_merge", new MergeVertex(), "stage2d_up", "stage1");
        RSU7(graph, "stage1d", "stage1_merge", 128, 16, 64);

        // Output -----------------------------
        graph
                .addLayer("stage1_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage1d")
//                .addLayer("stage1_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage1_out")
                .addLayer("stage2_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage2d")
                .addLayer("stage2_up", new Upsampling2D.Builder(2).build(), "stage2_out")
//                .addLayer("stage2_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage2_up")
                .addLayer("stage3_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(128)
                        .nOut(1)
                        .padding(1, 1).build(), "stage3d")
                .addLayer("stage3_up", new Upsampling2D.Builder(4).build(), "stage3_out")
//                .addLayer("stage3_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage3_up")
                .addLayer("stage4_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(256)
                        .nOut(1)
                        .padding(1, 1).build(), "stage4d")
                .addLayer("stage4_up", new Upsampling2D.Builder(8).build(), "stage4_out")
//                .addLayer("stage4_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage4_up")
                .addLayer("stage5_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(512)
                        .nOut(1)
                        .padding(1, 1).build(), "stage5d")
                .addLayer("stage5_up", new Upsampling2D.Builder(16).build(), "stage5_out")
//                .addLayer("stage5_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage5_up")
                .addLayer("stage6_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(512)
                        .nOut(1)
                        .padding(1, 1).build(), "stage5d")
                .addLayer("stage6_up", new Upsampling2D.Builder(16).build(), "stage6_out")
//                .addLayer("stage6_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage6_up")
                .addVertex("merge", new MergeVertex(), "stage1_out", "stage2_up", "stage3_up",
                        "stage4_up", "stage5_up", "stage6_up")
                .addLayer("output_conv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(6)
                        .nOut(1).build(), "merge")
                .addLayer("output", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).build(), "output_conv")
                .setOutputs("output")
        ;

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public static ComputationGraph initNETP(int width, int height, int depth) {
        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADADELTA)
                .weightInit(WeightInit.RELU)
                .l2(5e-5)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .graphBuilder();

        graph.addInputs("input").setInputTypes(InputType.convolutional(height, width, depth));

        RSU7(graph, "stage1", "input", 3, 16, 64);
        graph.addLayer("stage1_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage1");
        RSU6(graph, "stage2", "stage1_pool", 64, 16, 64);
        graph.addLayer("stage2_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage2");
        RSU5(graph, "stage3", "stage2_pool", 64, 16, 64);
        graph.addLayer("stage3_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage3");
        RSU4(graph, "stage4", "stage3_pool", 64, 16, 64);
        graph.addLayer("stage4_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage4");
        RSU4F(graph, "stage5", "stage4_pool", 64, 16, 64);
        graph.addLayer("stage5_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), "stage5");
        RSU4F(graph, "stage6", "stage5_pool", 64, 16, 64);
        // Decoder ---------------------------
        graph.addLayer("stage6d_up", new Upsampling2D.Builder(2).build(), "stage6");
        graph.addVertex("stage5_merge", new MergeVertex(), "stage6d_up", "stage5");
        RSU4F(graph, "stage5d", "stage5_merge", 128, 16, 64);
        graph.addLayer("stage5d_up", new Upsampling2D.Builder(2).build(), "stage5d");
        graph.addVertex("stage4_merge", new MergeVertex(), "stage5d_up", "stage4");
        RSU4(graph, "stage4d", "stage4_merge", 128, 16, 64);
        graph.addLayer("stage4d_up", new Upsampling2D.Builder(2).build(), "stage4d");
        graph.addVertex("stage3_merge", new MergeVertex(), "stage4d_up", "stage3");
        RSU5(graph, "stage3d", "stage3_merge", 128, 16, 64);
        graph.addLayer("stage3d_up", new Upsampling2D.Builder(2).build(), "stage3d");
        graph.addVertex("stage2_merge", new MergeVertex(), "stage3d_up", "stage2");
        RSU6(graph, "stage2d", "stage2_merge", 128, 16, 64);
        graph.addLayer("stage2d_up", new Upsampling2D.Builder(2).build(), "stage2d");
        graph.addVertex("stage1_merge", new MergeVertex(), "stage2d_up", "stage1");
        RSU7(graph, "stage1d", "stage1_merge", 128, 16, 64);

        // Output -----------------------------
        graph
                .addLayer("stage1_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage1d")
//                .addLayer("stage1_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage1_out")
                .addLayer("stage2_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage2d")
                .addLayer("stage2_up", new Upsampling2D.Builder(2).build(), "stage2_out")
//                .addLayer("stage2_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage2_up")
                .addLayer("stage3_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage3d")
                .addLayer("stage3_up", new Upsampling2D.Builder(4).build(), "stage3_out")
//                .addLayer("stage3_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage3_up")
                .addLayer("stage4_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage4d")
                .addLayer("stage4_up", new Upsampling2D.Builder(8).build(), "stage4_out")
//                .addLayer("stage4_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage4_up")
                .addLayer("stage5_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage5d")
                .addLayer("stage5_up", new Upsampling2D.Builder(16).build(), "stage5_out")
//                .addLayer("stage5_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage5_up")
                .addLayer("stage6_out", new ConvolutionLayer.Builder(3, 3)
                        .nIn(64)
                        .nOut(1)
                        .padding(1, 1).build(), "stage5d")
                .addLayer("stage6_up", new Upsampling2D.Builder(16).build(), "stage6_out")
//                .addLayer("stage6_over", new ActivationLayer.Builder()
//                        .activation(Activation.SIGMOID).build(), "stage6_up")
                .addVertex("merge", new MergeVertex(), "stage1_out", "stage2_up", "stage3_up",
                        "stage4_up", "stage5_up", "stage6_up")
                .addLayer("output_conv", new ConvolutionLayer.Builder(1, 1)
                        .nIn(6)
                        .nOut(1).build(), "merge")
                .addLayer("output", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).build(), "output_conv")
                .setOutputs("output")
        ;

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public static void ReBNConv(ComputationGraphConfiguration.GraphBuilder graph,
                                String name, String input, int inChannel, int outChannel, int dilation) {
        graph.addLayer(name + "_conv", new ConvolutionLayer.Builder(3, 3)
                .padding(dilation, dilation)
                .dilation(dilation, dilation)
                .nIn(inChannel)
                .nOut(outChannel).build(), input);
        graph.addLayer(name + "_bn", new BatchNormalization.Builder()
                .nIn(outChannel)
                .nOut(outChannel).build(), name + "_conv");
        graph.addLayer(name, new ActivationLayer.Builder()
                .activation(Activation.RELU).build(), name + "_bn");
    }

    public static void RSU4F(ComputationGraphConfiguration.GraphBuilder graph,
                             String name, String input, int inChannel, int midChannel, int outChannel) {
        ReBNConv(graph, name + "_in", input, inChannel, outChannel, 1);

        ReBNConv(graph, name + "_conv1", name + "_in", outChannel, midChannel, 1);
        ReBNConv(graph, name + "_conv2", name + "_conv1", midChannel, midChannel, 2);
        ReBNConv(graph, name + "_conv3", name + "_conv2", midChannel, midChannel, 4);

        ReBNConv(graph, name + "_conv4", name + "_conv3", midChannel, midChannel, 8);

        graph.addVertex(name + "merge43", new MergeVertex(), name + "_conv4", name + "_conv3");
        ReBNConv(graph, name + "_conv3d", name + "merge43", midChannel * 2, midChannel, 4);
        graph.addVertex(name + "merge32", new MergeVertex(), name + "_conv3d", name + "_conv2");
        ReBNConv(graph, name + "_conv2d", name + "merge32", midChannel * 2, midChannel, 2);
        graph.addVertex(name + "merge21", new MergeVertex(), name + "_conv2d", name + "_conv1");
        ReBNConv(graph, name + "_conv1d", name + "merge21", midChannel * 2, outChannel, 1);
        graph.addVertex(name, new MergeVertex(), name + "_conv1d", name + "_in");
    }

    public static void RSU4(ComputationGraphConfiguration.GraphBuilder graph,
                            String name, String input, int inChannel, int midChannel, int outChannel) {
        ReBNConv(graph, name + "_in", input, inChannel, outChannel, 1);

        ReBNConv(graph, name + "_conv1", name + "_in", outChannel, midChannel, 1);
        graph.addLayer(name + "_pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv1");
        ReBNConv(graph, name + "_conv2", name + "_pool1", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv2");

        ReBNConv(graph, name + "_conv3", name + "_pool2", midChannel, midChannel, 1);
        ReBNConv(graph, name + "_conv4", name + "_conv3", midChannel, midChannel, 2);

        graph.addVertex(name + "merge43", new MergeVertex(), name + "_conv4", name + "_conv3");
        ReBNConv(graph, name + "_conv3d", name + "merge43", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up3d", new Upsampling2D.Builder(2).build(), name + "_conv3d");
        graph.addVertex(name + "merge32", new MergeVertex(), name + "_up3d", name + "_conv2");
        ReBNConv(graph, name + "_conv2d", name + "merge32", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up2d", new Upsampling2D.Builder(2).build(), name + "_conv2d");
        graph.addVertex(name + "merge21", new MergeVertex(), name + "_up2d", name + "_conv1");
        ReBNConv(graph, name + "_conv1d", name + "merge21", midChannel * 2, outChannel, 1);

        graph.addVertex(name, new MergeVertex(), name + "_conv1d", name + "_in");
    }

    public static void RSU5(ComputationGraphConfiguration.GraphBuilder graph,
                            String name, String input, int inChannel, int midChannel, int outChannel) {
        ReBNConv(graph, name + "_in", input, inChannel, outChannel, 1);

        ReBNConv(graph, name + "_conv1", name + "_in", outChannel, midChannel, 1);
        graph.addLayer(name + "_pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv1");
        ReBNConv(graph, name + "_conv2", name + "_pool1", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv2");
        ReBNConv(graph, name + "_conv3", name + "_pool2", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv3");

        ReBNConv(graph, name + "_conv4", name + "_pool3", midChannel, midChannel, 1);
        ReBNConv(graph, name + "_conv5", name + "_conv4", midChannel, midChannel, 2);

        graph.addVertex(name + "merge54", new MergeVertex(), name + "_conv5", name + "_conv4");
        ReBNConv(graph, name + "_conv4d", name + "merge54", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up4d", new Upsampling2D.Builder(2).build(), name + "_conv4d");
        graph.addVertex(name + "merge43", new MergeVertex(), name + "_up4d", name + "_conv3");
        ReBNConv(graph, name + "_conv3d", name + "merge43", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up3d", new Upsampling2D.Builder(2).build(), name + "_conv3d");
        graph.addVertex(name + "merge32", new MergeVertex(), name + "_up3d", name + "_conv2");
        ReBNConv(graph, name + "_conv2d", name + "merge32", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up2d", new Upsampling2D.Builder(2).build(), name + "_conv2d");
        graph.addVertex(name + "merge21", new MergeVertex(), name + "_up2d", name + "_conv1");
        ReBNConv(graph, name + "_conv1d", name + "merge21", midChannel * 2, outChannel, 1);

        graph.addVertex(name, new MergeVertex(), name + "_conv1d", name + "_in");
    }

    public static void RSU6(ComputationGraphConfiguration.GraphBuilder graph,
                            String name, String input, int inChannel, int midChannel, int outChannel) {
        ReBNConv(graph, name + "_in", input, inChannel, outChannel, 1);

        ReBNConv(graph, name + "_conv1", name + "_in", outChannel, midChannel, 1);
        graph.addLayer(name + "_pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv1");
        ReBNConv(graph, name + "_conv2", name + "_pool1", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv2");
        ReBNConv(graph, name + "_conv3", name + "_pool2", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv3");
        ReBNConv(graph, name + "_conv4", name + "_pool3", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv4");

        ReBNConv(graph, name + "_conv5", name + "_pool4", midChannel, midChannel, 1);
        ReBNConv(graph, name + "_conv6", name + "_conv5", midChannel, midChannel, 2);

        graph.addVertex(name + "merge65", new MergeVertex(), name + "_conv6", name + "_conv5");
        ReBNConv(graph, name + "_conv5d", name + "merge65", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up5d", new Upsampling2D.Builder(2).build(), name + "_conv5d");
        graph.addVertex(name + "merge54", new MergeVertex(), name + "_up5d", name + "_conv4");
        ReBNConv(graph, name + "_conv4d", name + "merge54", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up4d", new Upsampling2D.Builder(2).build(), name + "_conv4d");
        graph.addVertex(name + "merge43", new MergeVertex(), name + "_up4d", name + "_conv3");
        ReBNConv(graph, name + "_conv3d", name + "merge43", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up3d", new Upsampling2D.Builder(2).build(), name + "_conv3d");
        graph.addVertex(name + "merge32", new MergeVertex(), name + "_up3d", name + "_conv2");
        ReBNConv(graph, name + "_conv2d", name + "merge32", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up2d", new Upsampling2D.Builder(2).build(), name + "_conv2d");
        graph.addVertex(name + "merge21", new MergeVertex(), name + "_up2d", name + "_conv1");
        ReBNConv(graph, name + "_conv1d", name + "merge21", midChannel * 2, outChannel, 1);

        graph.addVertex(name, new MergeVertex(), name + "_conv1d", name + "_in");
    }

    public static void RSU7(ComputationGraphConfiguration.GraphBuilder graph,
                            String name, String input, int inChannel, int midChannel, int outChannel) {
        ReBNConv(graph, name + "_in", input, inChannel, outChannel, 1);

        ReBNConv(graph, name + "_conv1", name + "_in", outChannel, midChannel, 1);
        graph.addLayer(name + "_pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv1");
        ReBNConv(graph, name + "_conv2", name + "_pool1", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv2");
        ReBNConv(graph, name + "_conv3", name + "_pool2", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv3");
        ReBNConv(graph, name + "_conv4", name + "_pool3", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv4");
        ReBNConv(graph, name + "_conv5", name + "_pool4", midChannel, midChannel, 1);
        graph.addLayer(name + "_pool5", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build(), name + "_conv5");

        ReBNConv(graph, name + "_conv6", name + "_pool5", midChannel, midChannel, 1);
        ReBNConv(graph, name + "_conv7", name + "_conv6", midChannel, midChannel, 2);

        graph.addVertex(name + "merge76", new MergeVertex(), name + "_conv7", name + "_conv6");
        ReBNConv(graph, name + "_conv6d", name + "merge76", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up6d", new Upsampling2D.Builder(2).build(), name + "_conv6d");
        graph.addVertex(name + "merge65", new MergeVertex(), name + "_up6d", name + "_conv5");
        ReBNConv(graph, name + "_conv5d", name + "merge65", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up5d", new Upsampling2D.Builder(2).build(), name + "_conv5d");
        graph.addVertex(name + "merge54", new MergeVertex(), name + "_up5d", name + "_conv4");
        ReBNConv(graph, name + "_conv4d", name + "merge54", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up4d", new Upsampling2D.Builder(2).build(), name + "_conv4d");
        graph.addVertex(name + "merge43", new MergeVertex(), name + "_up4d", name + "_conv3");
        ReBNConv(graph, name + "_conv3d", name + "merge43", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up3d", new Upsampling2D.Builder(2).build(), name + "_conv3d");
        graph.addVertex(name + "merge32", new MergeVertex(), name + "_up3d", name + "_conv2");
        ReBNConv(graph, name + "_conv2d", name + "merge32", midChannel * 2, midChannel, 1);
        graph.addLayer(name + "_up2d", new Upsampling2D.Builder(2).build(), name + "_conv2d");
        graph.addVertex(name + "merge21", new MergeVertex(), name + "_up2d", name + "_conv1");
        ReBNConv(graph, name + "_conv1d", name + "merge21", midChannel * 2, outChannel, 1);

        graph.addVertex(name, new MergeVertex(), name + "_conv1d", name + "_in");
    }
}
