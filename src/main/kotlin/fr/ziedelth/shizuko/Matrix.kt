package fr.ziedelth.shizuko

import java.util.*
import kotlin.math.pow

data class Matrix(
    private val rows: Int,
    private val columns: Int,
    var data: DoubleArray = DoubleArray(rows * columns),
) {
    @Transient
    val columnInversed: Double = 1.0 / columns

    fun get(x: Int, y: Int): Double {
        return data[x * columns + y]
    }

    fun set(x: Int, y: Int, value: Double) {
        data[x * columns + y] = value
    }

    fun map(function: (Double, Int, Int) -> Double): Matrix {
        for (index in data.indices) {
            val x = (index * columnInversed).toInt()
            val y = index % columns
            data[index] = function(data[index], x, y)
        }

        return this
    }

    fun randomize(random: Random, min: Double = -1.0, max: Double = 1.0): Matrix {
        return map { _, _, _ -> random.nextDouble() * (max - min) + min }
    }

    fun add(other: Matrix): Matrix {
        return map { `val`, x, y -> `val` + other.get(x, y) }
    }

    fun subtract(other: Matrix): Matrix {
        return map { `val`, x, y -> `val` - other.get(x, y) }
    }

    fun transpose(): Matrix {
        val result = Matrix(columns, rows)
        result.map { _, x, y -> get(y, x) }
        return result
    }

    fun elementMult(other: Matrix): Matrix {
        return map { `val`, x, y -> `val` * other.get(x, y) }
    }

    fun multiply(other: Matrix): Matrix {
        return Matrix(rows, other.columns).map { _, x, y ->
            var sum = 0.0
            for (i in 0 until columns) {
                sum += get(x, i) * other.get(i, y)
            }
            sum
        }
    }

    fun scale(scalar: Double): Matrix {
        return map { `val`, _, _ -> `val` * scalar }
    }

    fun toArray(): DoubleArray {
        val result = DoubleArray(rows * columns)
        System.arraycopy(data, 0, result, 0, rows * columns)
        return result
    }

    fun copy(): Matrix {
        return Matrix(rows, columns, data.copyOf())
    }

    fun pow(power: Double): Matrix {
        return map { `val`, _, _ -> `val`.pow(power) }
    }

    fun mean(): Double {
        return data.sum() / data.size
    }

    companion object {
        fun fromArray(array: DoubleArray): Matrix {
            val result = Matrix(array.size, 1)
            System.arraycopy(array, 0, result.data, 0, array.size)
            return result
        }
    }
}
