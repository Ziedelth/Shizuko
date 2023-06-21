package fr.ziedelth.shizuko

import com.google.gson.Gson
import fr.ziedelth.shizuko.Parallelism.parallelForEach
import fr.ziedelth.shizuko.utils.Chart
import javafx.scene.chart.XYChart
import java.io.File
import kotlin.system.measureTimeMillis

data class Data(
    val inputs: DoubleArray,
    val outputs: DoubleArray,
)

data class Dataset(
    val trainingSet: Collection<Data>,
    val testSet: Collection<Data>,
) {
    private fun drawProgressbar(progress: Double, length: Int = 50, redraw: Boolean = true, remainingTimeMs: Double? = null) {
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

    fun getRandomBatch(batchSize: Int): List<Collection<Data>> {
        return trainingSet.shuffled().chunked(batchSize)
    }

    fun train(neuralNetwork: NeuralNetwork, batchSize: Int, epochs: Int) {
        val chart = Chart("Epochs", "Average")

        val avgLossSeries = XYChart.Series<Number, Number>()
        avgLossSeries.name = "Average loss"

        val times = mutableListOf<Long>()

        for (i in 1..epochs) {
            val losses = mutableListOf<Double>()

            times.add(measureTimeMillis {
                for (batch in getRandomBatch(batchSize)) {
                    for (data in batch) {
                        losses.add(neuralNetwork.train(data.inputs, data.outputs))
                    }
                }
            })

            val remainingTimeMs = (epochs - i) * times.average()
            drawProgressbar(i.toDouble() / epochs, remainingTimeMs = remainingTimeMs)

            if (i % 10 == 0 && avgLossSeries.data.none { xy -> xy.xValue.toInt() == i }) {
                try {
                    avgLossSeries.data.add(XYChart.Data(i, losses.average()))
                    chart.save(File("chart.png"), avgLossSeries)
                } catch (e: Exception) {
                    println("Couldn't save chart: ${e.message}")
                }
            }
        }

        chart.close()
    }

    companion object {
        fun load(file: File): Dataset {
            return Gson().fromJson(file.readText(), Dataset::class.java)
        }
    }
}
