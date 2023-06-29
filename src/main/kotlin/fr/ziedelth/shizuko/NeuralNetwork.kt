package fr.ziedelth.shizuko

import com.google.gson.Gson
import java.io.File
import java.util.*
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.roundToInt

private interface IFunction {
    fun apply(matrix: Matrix): Matrix
    fun applyDerivative(matrix: Matrix): Matrix
}

private class Sigmoid : IFunction {
    override fun apply(matrix: Matrix): Matrix {
        return matrix.map { `val`, _, _ -> 1 / (1 + exp(-`val`)) }
    }

    override fun applyDerivative(matrix: Matrix): Matrix {
        return matrix.map { `val`, _, _ -> `val` * (1 - `val`) }
    }
}

private val functions = mapOf(
    "sigmoid" to Sigmoid(),
)

data class NeuralNetwork(
    private val inputs: Int,
    private val hiddenLayers: Int,
    private val hidden: Int,
    private val outputs: Int,
    private val learningRate: Double = 0.001,
    private val activationFunction: String = "sigmoid",
) {
    @Transient
    private val random = Random()
    @Transient
    private val function: IFunction = functions[activationFunction] ?: error("Unknown activation function: $activationFunction")

    private val weights: Array<Matrix> = Array(hiddenLayers + 1) {
        when (it) {
            0 -> {
                Matrix(hidden, inputs).randomize(random)
            }
            hiddenLayers -> {
                Matrix(outputs, hidden).randomize(random)
            }
            else -> {
                Matrix(hidden, hidden).randomize(random)
            }
        }
    }
    private val biases: Array<Matrix> = Array(hiddenLayers + 1) {
        Matrix(if (it == hiddenLayers) outputs else hidden, 1).randomize(random)
    }

    init {
        require(inputs > 0) { "Inputs must be greater than 0" }
        require(hiddenLayers > 0) { "Hidden layers must be greater than 0" }
        require(hidden > 0) { "Hidden must be greater than 0" }
        require(outputs > 0) { "Outputs must be greater than 0" }
        require(learningRate > 0) { "Learning rate must be greater than 0" }
        require(activationFunction in functions.keys) { "Unknown activation function: $activationFunction" }
    }

    private fun calculateLayer(weights: Matrix, biases: Matrix, inputs: Matrix): Matrix {
        return function.apply(weights.multiply(inputs).add(biases))
    }

    private fun calculateGradient(output: Matrix, target: Matrix): Matrix {
        return function.applyDerivative(output).elementMult(target.subtract(output)).scale(learningRate)
    }

    private fun calculateDeltas(gradient: Matrix, previousOutput: Matrix): Matrix {
        return gradient.multiply(previousOutput.transpose())
    }

    fun feedForward(inputArray: DoubleArray): DoubleArray {
        var inputs = Matrix.fromArray(inputArray)
        for (i in 0 until hiddenLayers + 1) {
            inputs = calculateLayer(weights[i], biases[i], inputs)
        }
        return inputs.toArray()
    }

    fun meanSquaredError(actual: DoubleArray, expected: DoubleArray): Double {
        var sum = 0.0
        for (i in actual.indices) {
            sum += (actual[i] - expected[i]).pow(2.0)
        }
        return 1.0 / actual.size * sum
    }

    fun accuracy(actual: DoubleArray, expected: DoubleArray): Double {
        var correct = 0
        // actual can be between 0 and 1, so we round it to 0 or 1
        for (i in actual.indices) {
            if (actual[i].roundToInt() == expected[i].roundToInt()) {
                correct++
            }
        }
        return correct.toDouble() / actual.size
    }

    fun train(inputArray: DoubleArray, targetArray: DoubleArray) {
        var inputs = Matrix.fromArray(inputArray)
        var targets = Matrix.fromArray(targetArray)

        val layers = arrayOfNulls<Matrix>(hiddenLayers + 2)
        layers[0] = inputs

        for (i in 1 until hiddenLayers + 2) {
            layers[i] = calculateLayer(weights[i - 1], biases[i - 1], inputs)
            inputs = layers[i]!!
        }

        for (i in (hiddenLayers + 1) downTo 1) {
            val errors = targets.subtract(layers[i]!!)
            val gradients = calculateGradient(layers[i]!!, errors)
            val deltas = calculateDeltas(gradients, layers[i - 1]!!)
            biases[i - 1] = biases[i - 1].add(gradients)
            weights[i - 1] = weights[i - 1].add(deltas)
            val previousError = weights[i - 1].transpose().multiply(errors)
            targets = previousError
        }
    }

    fun save(file: File) {
        file.writeText(Gson().toJson(this))
    }

    companion object {
        fun load(file: File): NeuralNetwork {
            return Gson().fromJson(file.readText(), NeuralNetwork::class.java)
        }
    }
}
