package ai.certifai.training.feedforward;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class MnistFNN {

    final static int seed = 1234;
    final static int batchSize = 500;
    final static int epoch = 1;

    public static void main(String[] args) throws IOException {

        MnistDataSetIterator trainMnist = new MnistDataSetIterator(batchSize, true, seed);
        MnistDataSetIterator testMnist = new MnistDataSetIterator(batchSize, false, seed);

        //normalisation
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(trainMnist);
        trainMnist.setPreProcessor(scaler);
        testMnist.setPreProcessor(scaler);

        //model config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainMnist.inputColumns())
                        .nOut(124)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(282)
                        .build())
                .layer(new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(trainMnist.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000));

        for (int i = 0; i <= epoch; i++) {
            model.fit(trainMnist);
        }

        Evaluation evalTrain = model.evaluate(trainMnist);
        Evaluation evalTest = model.evaluate(testMnist);

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());

    }

}

// ========================Evaluation Metrics========================
//  # of classes:    10
//  Accuracy:        0.9526
//  Precision:       0.9534
//  Recall:          0.9520
//  F1 Score:        0.9523
// Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)


// =========================Confusion Matrix=========================
//     0    1    2    3    4    5    6    7    8    9
// ---------------------------------------------------
//  5801    1   16    9   11    9   20    1   33   22 | 0 = 0
//     1 6619   27   23    9    7    2    9   21   24 | 1 = 1
//    12   28 5692   73   41    6   10   49   32   15 | 2 = 2
//     8   24   74 5800    6   71    4   42   36   66 | 3 = 3
//    10   16   25    5 5454    1   24    9    9  289 | 4 = 4
//    30   16   15  111   30 5085   39   10   29   56 | 5 = 5
//    38   15   18    4   39   61 5725    1   16    1 | 6 = 6
//    12   23   37   12   29    2    1 5989    4  156 | 7 = 7
//    25   78   26  182   23   51   25   11 5270  160 | 8 = 8
//    21    8    1   63   66   11    1   44   12 5722 | 9 = 9

// Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
// ==================================================================


// ========================Evaluation Metrics========================
//  # of classes:    10
//  Accuracy:        0.9496
//  Precision:       0.9504
//  Recall:          0.9488
//  F1 Score:        0.9492
// Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)


// =========================Confusion Matrix=========================
//     0    1    2    3    4    5    6    7    8    9
// ---------------------------------------------------
//   968    0    1    1    0    3    3    1    1    2 | 0 = 0
//     0 1121    3    1    1    1    3    2    3    0 | 1 = 1
//     4    1  990   13    3    0    4    8    6    3 | 2 = 2
//     0    0   14  957    1    9    0   10    7   12 | 3 = 3
//     2    0    8    1  922    0    3    2    2   42 | 4 = 4
//     6    2    0   31    4  825    8    2    7    7 | 5 = 5
//    12    3    2    1    6   12  920    1    1    0 | 6 = 6
//     1    8   15    7    3    1    0  959    0   34 | 7 = 7
//     4    4    5   35    8   10    9    7  871   21 | 8 = 8
//     8    6    2   14   11    1    0    4    0  963 | 9 = 9

// Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
// ==================================================================
