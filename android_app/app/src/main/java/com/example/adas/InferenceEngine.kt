package com.example.adas

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Handles on-device inference using TensorFlow Lite.
 *
 * To use a real model:
 *  1. Export your YOLOv8 model: `yolo export model=yolov8n.pt format=tflite imgsz=320`
 *  2. Place the exported `yolov8n_float32.tflite` (or int8) in `app/src/main/assets/`
 *  3. Update [MODEL_FILENAME] and [INPUT_SIZE] accordingly.
 *  4. Replace the mock output in [detect] with real TFLite interpreter calls.
 */
class InferenceEngine(private val context: Context) {

    companion object {
        private const val MODEL_FILENAME = "yolov8n.tflite" // Place model in assets/
        private const val INPUT_SIZE = 320
        private const val CONFIDENCE_THRESHOLD = 0.4f
    }

    // TFLite interpreter — uncomment once a real model is added to assets
    // private val interpreter: Interpreter by lazy {
    //     val model = FileUtil.loadMappedFile(context, MODEL_FILENAME)
    //     val options = Interpreter.Options().apply { numThreads = 4 }
    //     Interpreter(model, options)
    // }

    private var frameCount = 0L

    /**
     * Runs inference on the provided [bitmap] and returns a list of [Detection] results.
     * Runs on [Dispatchers.Default] (background thread) to avoid blocking the UI.
     *
     * Currently returns deterministic mock bounding boxes so you can verify
     * the overlay rendering before adding a real TFLite model.
     */
    suspend fun detect(bitmap: Bitmap): List<Detection> = withContext(Dispatchers.Default) {
        frameCount++

        // ── MOCK OUTPUT ─────────────────────────────────────────────────────────
        // Replace this entire block with real TFLite interpreter calls once a
        // .tflite model file is placed in the assets/ directory.
        val mockDetections = mutableListOf<Detection>()

        // Simulate a "vehicle ahead" detection that slowly drifts across the frame
        val t = (frameCount % 120) / 120f
        val cx = 0.3f + t * 0.4f // 0.3 → 0.7 sweep
        val cy = 0.45f + kotlin.math.sin(t * Math.PI.toFloat()) * 0.08f

        mockDetections.add(
            Detection(
                label = "vehicle",
                confidence = 0.87f,
                boundingBox = RectF(cx - 0.12f, cy - 0.10f, cx + 0.12f, cy + 0.10f)
            )
        )

        // Simulate a pedestrian on the left side of the frame
        if (frameCount % 60 < 45) {
            mockDetections.add(
                Detection(
                    label = "person",
                    confidence = 0.73f,
                    boundingBox = RectF(0.05f, 0.35f, 0.18f, 0.70f)
                )
            )
        }

        // Simulate a two-wheeler on the right
        mockDetections.add(
            Detection(
                label = "motorcycle",
                confidence = 0.61f,
                boundingBox = RectF(0.72f, 0.50f, 0.90f, 0.78f)
            )
        )
        // ── END MOCK OUTPUT ──────────────────────────────────────────────────────

        mockDetections.filter { it.confidence >= CONFIDENCE_THRESHOLD }
    }
}
