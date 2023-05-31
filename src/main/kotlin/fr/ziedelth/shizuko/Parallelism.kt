package fr.ziedelth.shizuko

import java.util.concurrent.Callable
import java.util.concurrent.Executors

object Parallelism {
    private val processors = Runtime.getRuntime().availableProcessors() - 1
    private val executor = Executors.newFixedThreadPool(processors)

    fun <T> Iterable<T>.parallelForEachChunked(action: (T) -> Unit) {
        map { Callable { action(it) } }.chunked(processors).forEach { executor.invokeAll(it) }
    }

    fun <T> Iterable<T>.parallelForEach(action: (T) -> Unit) {
        executor.invokeAll(map { Callable { action(it) } })
    }

    fun stop() {
        executor.shutdown()
    }
}