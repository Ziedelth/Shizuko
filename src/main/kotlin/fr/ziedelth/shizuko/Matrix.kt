package fr.ziedelth.shizuko

import java.util.*

data class Matrix(
    private val rows: Int,
    private val columns: Int,
    private val data: DoubleArray = DoubleArray(rows * columns),
) {
    @Transient
    private val random = Random()

    fun get(x: Int, y: Int): Double {
        return data[x * columns + y]
    }

    fun set(x: Int, y: Int, value: Double) {
        data[x * columns + y] = value
    }

    fun map(function: (Double, Int, Int) -> Double): Matrix {
        data.indices.forEach { index ->
            val x = index / columns
            val y = index % columns
            data[index] = function(data[index], x, y)
        }

        return this
    }

    fun randomize(min: Double = -1.0, max: Double = 1.0): Matrix {
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
            (0 until columns).forEach { i -> sum += get(x, i) * other.get(i, y) }
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

    companion object {
        fun fromArray(array: DoubleArray): Matrix {
            val result = Matrix(array.size, 1)
            System.arraycopy(array, 0, result.data, 0, array.size)
            return result
        }
    }
}
