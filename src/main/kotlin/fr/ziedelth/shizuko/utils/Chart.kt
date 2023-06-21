package fr.ziedelth.shizuko.utils

import javafx.application.Platform
import javafx.embed.swing.JFXPanel
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Scene
import javafx.scene.chart.LineChart
import javafx.scene.chart.NumberAxis
import javafx.scene.chart.XYChart
import javafx.scene.image.WritableImage
import javafx.stage.Stage
import java.io.File
import javax.imageio.ImageIO

class Chart(
    xAxisName: String,
    yAxisName: String,
) {
    private var xAxis: NumberAxis
    private var yAxis: NumberAxis

    init {
        JFXPanel()
        xAxis = NumberAxis()
        xAxis.label = xAxisName
        yAxis = NumberAxis()
        yAxis.label = yAxisName
    }

    fun save(file: File, vararg series: XYChart.Series<Number, Number>) {
        val lineChart = LineChart(xAxis, yAxis)
        lineChart.animated = false
        lineChart.createSymbols = false
        lineChart.data.addAll(series)

        Platform.runLater {
            val stage = Stage()
            val scene = Scene(lineChart, 800.0, 600.0)
            stage.scene = scene

            try {
                val writableImage = WritableImage(800, 600)
                scene.snapshot(writableImage)
                ImageIO.write(SwingFXUtils.fromFXImage(writableImage, null), "png", file)
            } catch (_: Exception) {
                // Ignore
            }

            stage.close()
        }
    }

    fun close() {
        Platform.exit()
    }
}