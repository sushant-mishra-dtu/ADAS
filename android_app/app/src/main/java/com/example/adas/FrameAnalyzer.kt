package com.example.adas

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
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
) : ImageAnalysis.Analyzer {

    // A supervised scope so one failed frame doesn't cancel the entire pipeline
    private val analyzerScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun analyze(image: ImageProxy) {
        // Use CameraX's built-in toBitmap() extension — closes the image internally
        val bitmap: Bitmap? = try {
            image.toBitmap()
        } catch (e: Exception) {
            null
        } finally {
            image.close()
        }

        if (bitmap != null) {
            analyzerScope.launch {
                val results = engine.detect(bitmap)
                withContext(Dispatchers.Main) {
                    onResults(results)
                }
            }
        }
    }
}
