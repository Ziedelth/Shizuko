package fr.ziedelth.shizuko

import com.google.gson.Gson
import fr.ziedelth.shizuko.utils.Chart
import javafx.scene.chart.XYChart
import org.ejml.simple.SimpleMatrix
import java.io.File
import kotlin.system.measureTimeMillis

data class Data(
    val inputs: DoubleArray,
    val outputs: DoubleArray,
) {
    @Transient
    lateinit var inputMatrix: SimpleMatrix

    @Transient
    lateinit var outputMatrix: SimpleMatrix

    fun loadMatrices() {
        inputMatrix = toMatrix(inputs)
        outputMatrix = toMatrix(outputs)
    }
}

data class Dataset(
    val trainingSet: Collection<Data>,
    val testSet: Collection<Data>,
) {
    private fun drawProgressbar(
        progress: Double,
        length: Int = 50,
        redraw: Boolean = true,
        remainingTimeMs: Double? = null
    ) {
        val copyProgressBar = " ".repeat(length).toCharArray()
        val currentPosition = (copyProgressBar.size * progress).toInt()
        for (i in 0 until currentPosition) copyProgressBar[i] = '•'
        val str = "|${copyProgressBar.joinToString("")}|\t${String.format("%.2f", progress * 100)}%${
            if (remainingTimeMs != null) "\t~${
                String.format(
                    "%.2f",
                    remainingTimeMs / 60000.0
                )
            } min (${(remainingTimeMs / 1000).toInt()} s)" else ""
        }"
        if (redraw) print("\b".repeat(str.length + 1) + str) else println(str)
    }

    fun save(file: File) {
        file.writeText(Gson().toJson(this))
    }

    fun train(neuralNetwork: NeuralNetwork, epochs: Int) {
        val lossChart = Chart("Epochs", "Average loss")
        val accuracyChart = Chart("Epochs", "Accuracy")

        val avgTrainingLossSeries = XYChart.Series<Number, Number>()
        avgTrainingLossSeries.name = "Training set"
        val avgTestLossSeries = XYChart.Series<Number, Number>()
        avgTestLossSeries.name = "Test set"

        val avgTrainingAccuracySeries = XYChart.Series<Number, Number>()
        avgTrainingAccuracySeries.name = "Training set"
        val avgTestAccuracySeries = XYChart.Series<Number, Number>()
        avgTestAccuracySeries.name = "Test set"

        val times = mutableListOf<Long>()

        for (i in 1..epochs) {
            val evaluate = i == 1 || i % 10 == 0

            val trainingLosses = mutableListOf<Double>()
            val testLosses = mutableListOf<Double>()
            val trainingAccuracies = mutableListOf<Double>()
            val testAccuracies = mutableListOf<Double>()

            times.add(measureTimeMillis {
                for (data in trainingSet) {
                    neuralNetwork.train(data.inputMatrix, data.outputMatrix)
                }

                if (evaluate) {
                    // Evaluate the model
                    for (data in trainingSet) {
                        val predicted = neuralNetwork.feedForward(data.inputs)
                        trainingLosses.add(neuralNetwork.meanSquaredError(predicted, data.outputs))
                        trainingAccuracies.add(neuralNetwork.accuracy(predicted, data.outputs))
                    }

                    for (data in testSet) {
                        val predictedOutputs = neuralNetwork.feedForward(data.inputs)
                        testLosses.add(neuralNetwork.meanSquaredError(predictedOutputs, data.outputs))
                        testAccuracies.add(neuralNetwork.accuracy(predictedOutputs, data.outputs))
                    }
                }
            })

            val remainingTimeMs = (epochs - i) * times.average()
            drawProgressbar(i.toDouble() / epochs, remainingTimeMs = remainingTimeMs)

            if (evaluate) {
                try {
                    avgTrainingLossSeries.data.add(XYChart.Data(i, trainingLosses.average()))
                    avgTestLossSeries.data.add(XYChart.Data(i, testLosses.average()))
                    lossChart.save(File("loss-chart.png"), avgTrainingLossSeries, avgTestLossSeries)

                    avgTrainingAccuracySeries.data.add(XYChart.Data(i, trainingAccuracies.average()))
                    avgTestAccuracySeries.data.add(XYChart.Data(i, testAccuracies.average()))
                    accuracyChart.save(File("accuracy-chart.png"), avgTrainingAccuracySeries, avgTestAccuracySeries)
                } catch (e: Exception) {
                    println("Couldn't save chart: ${e.message}")
                }
            }
        }

        lossChart.close()
    }

    companion object {
        fun load(file: File): Dataset {
            val data = Gson().fromJson(file.readText(), Dataset::class.java)
            data.trainingSet.forEach { it.loadMatrices() }
            data.testSet.forEach { it.loadMatrices() }
            return data
        }
    }
}
