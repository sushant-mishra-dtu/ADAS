package com.example.adas

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.Closeable
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * CameraX [ImageAnalysis.Analyzer] that extracts frames from the camera pipeline
 * and passes them to the [InferenceEngine] on a background coroutine.
 *
 * Results are delivered via the [onResults] callback on the **main** thread,
 * ready to drive a recomposition of the overlay.
 */
class FrameAnalyzer(
    private val engine: InferenceEngine,
    private val onResults: (List<Detection>) -> Unit
) : ImageAnalysis.Analyzer, Closeable {

    // A supervised scope so one failed frame doesn't cancel the entire pipeline
    private val analyzerJob: Job = SupervisorJob()
    private val analyzerScope = CoroutineScope(analyzerJob + Dispatchers.Default)
    private var pendingJob: Job? = null

    override fun analyze(image: ImageProxy) {
        if (pendingJob?.isActive == true) {
            image.close()
            return
        }

        // Use CameraX's built-in toBitmap() extension — closes the image internally
        val bitmap: Bitmap? = try {
            image.toBitmap()
        } catch (e: Exception) {
            null
        } finally {
            image.close()
        }

        if (bitmap != null) {
            pendingJob = analyzerScope.launch {
                val results = engine.detect(bitmap)
                withContext(Dispatchers.Main) {
                    onResults(results)
                }
            }
        }
    }

    override fun close() {
        analyzerScope.cancel()
    }
}
