package fr.ziedelth.shizuko

import com.google.gson.Gson
import org.ejml.simple.SimpleMatrix
import java.io.File
import java.util.*
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.roundToInt

fun toMatrix(array: DoubleArray): SimpleMatrix {
    return SimpleMatrix(arrayOf(array)).transpose()
}

private interface IFunction {
    fun apply(matrix: SimpleMatrix): SimpleMatrix
    fun applyDerivative(matrix: SimpleMatrix): SimpleMatrix
}

private class Sigmoid : IFunction {
    override fun apply(matrix: SimpleMatrix): SimpleMatrix {
        for (i in 0 until matrix.numRows) {
            for (j in 0 until matrix.numCols) {
                matrix[i, j] = 1 / (1 + exp(-matrix[i, j]))
            }
        }

        return matrix
    }

    override fun applyDerivative(matrix: SimpleMatrix): SimpleMatrix {
        for (i in 0 until matrix.numRows) {
            for (j in 0 until matrix.numCols) {
                val value = matrix[i, j]
                matrix[i, j] = value * (1 - value)
            }
        }

        return matrix
    }
}

private class Relu : IFunction {
    override fun apply(matrix: SimpleMatrix): SimpleMatrix {
        for (i in 0 until matrix.numRows) {
            for (j in 0 until matrix.numCols) {
                matrix[i, j] = if (matrix[i, j] > 0) matrix[i, j] else 0.0
            }
        }

        return matrix
    }

    override fun applyDerivative(matrix: SimpleMatrix): SimpleMatrix {
        for (i in 0 until matrix.numRows) {
            for (j in 0 until matrix.numCols) {
                matrix[i, j] = if (matrix[i, j] > 0) 1.0 else 0.0
            }
        }

        return matrix
    }
}

private val functions = mapOf(
    "sigmoid" to Sigmoid(),
    "relu" to Relu(),
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
    private val function: IFunction =
        functions[activationFunction] ?: error("Unknown activation function: $activationFunction")

    private val weights: Array<SimpleMatrix> = Array(hiddenLayers + 1) {
        when (it) {
            0 -> {
                SimpleMatrix.random_DDRM(hidden, inputs, -1.0, 1.0, random)
            }

            hiddenLayers -> {
                SimpleMatrix.random_DDRM(outputs, hidden, -1.0, 1.0, random)
            }

            else -> {
                SimpleMatrix.random_DDRM(hidden, hidden, -1.0, 1.0, random)
            }
        }
    }
    private val biases: Array<SimpleMatrix> = Array(hiddenLayers + 1) {
        if (it == hiddenLayers) {
            SimpleMatrix.random_DDRM(outputs, 1, -1.0, 1.0, random)
        } else {
            SimpleMatrix.random_DDRM(hidden, 1, -1.0, 1.0, random)
        }
    }

    init {
        require(inputs > 0) { "Inputs must be greater than 0" }
        require(hiddenLayers > 0) { "Hidden layers must be greater than 0" }
        require(hidden > 0) { "Hidden must be greater than 0" }
        require(outputs > 0) { "Outputs must be greater than 0" }
        require(learningRate > 0) { "Learning rate must be greater than 0" }
        require(activationFunction in functions.keys) { "Unknown activation function: $activationFunction" }
    }

    private fun toArray(matrix: SimpleMatrix, column: Int): DoubleArray {
        val array = DoubleArray(matrix.numRows)
        for (i in 0 until matrix.numRows) {
            array[i] = matrix[i, column]
        }
        return array
    }

    private fun calculateLayer(weights: SimpleMatrix, biases: SimpleMatrix, inputs: SimpleMatrix): SimpleMatrix {
        return function.apply(weights.mult(inputs).plus(biases))
    }

    private fun calculateGradient(output: SimpleMatrix, error: SimpleMatrix): SimpleMatrix {
        return function.applyDerivative(output).elementMult(error).scale(learningRate)
    }

    private fun calculateDeltas(gradient: SimpleMatrix, previousOutput: SimpleMatrix): SimpleMatrix {
        return gradient.mult(previousOutput.transpose())
    }

    fun feedForward(inputArray: DoubleArray): DoubleArray {
        var inputs = toMatrix(inputArray)
        for (i in 0 until hiddenLayers + 1) {
            inputs = calculateLayer(weights[i], biases[i], inputs)
        }
        return toArray(inputs, 0)
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

    fun train(inputMatrix: SimpleMatrix, targetMatrix: SimpleMatrix) {
        var inputs = inputMatrix.copy()
        var targets = targetMatrix.copy()

        val layers = arrayOfNulls<SimpleMatrix>(hiddenLayers + 2)
        layers[0] = inputs

        for (i in 1 until hiddenLayers + 2) {
            layers[i] = calculateLayer(weights[i - 1], biases[i - 1], inputs)
            inputs = layers[i]!!
        }

        for (i in (hiddenLayers + 1) downTo 1) {
            val errors = targets.minus(layers[i]!!)
            val gradients = calculateGradient(layers[i]!!, errors)
            val deltas = calculateDeltas(gradients, layers[i - 1]!!)
            biases[i - 1] = biases[i - 1].plus(gradients)
            weights[i - 1] = weights[i - 1].plus(deltas)
            val previousError = weights[i - 1].transpose().mult(errors)
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
