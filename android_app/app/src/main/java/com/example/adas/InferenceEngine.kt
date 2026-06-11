package com.example.adas

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * On-device YOLOv8n inference using TensorFlow Lite.
 *
 * Setup requirements:
 *  1. Export model: `yolo export model=yolov8n.pt format=tflite imgsz=320`
 *  2. Place `yolov8n.tflite` in app/src/main/assets/
 *  3. Place `coco_labels.txt` in app/src/main/assets/
 *
 * YOLOv8n TFLite output layout: [1, 84, 8400]
 *   - Rows 0–3    : cx, cy, w, h  (pixel coords relative to INPUT_SIZE)
 *   - Rows 4–83   : class scores  (already sigmoided by TFLite export)
 *   - Columns     : 8400 anchor predictions
 */
class InferenceEngine(private val context: Context) {

    companion object {
        private const val MODEL_FILENAME       = "yolov8n.tflite"
        private const val LABELS_FILENAME      = "coco_labels.txt"
        private const val INPUT_SIZE           = 320
        private const val CONFIDENCE_THRESHOLD = 0.40f
        private const val IOU_THRESHOLD        = 0.45f
        private const val NUM_CLASSES          = 80
        // Anchors for imgsz=320: feature maps 40×40 + 20×20 + 10×10 = 2100
        // (Would be 8400 for imgsz=640: 80×80 + 40×40 + 20×20)
        private const val NUM_ANCHORS          = 2100
    }

    // Lazy-loaded — model file is memory-mapped from assets on first use
    private val interpreter: Interpreter by lazy {
        val model   = loadModelFile(context, MODEL_FILENAME)
        val options = Interpreter.Options().apply { numThreads = 4 }
        Interpreter(model, options)
    }

    /**
     * Loads a TFLite model from the app's assets folder as a [MappedByteBuffer].
     */
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // 80 COCO class names loaded once from assets
    private val labels: List<String> by lazy {
        context.assets.open(LABELS_FILENAME)
            .let { BufferedReader(InputStreamReader(it)) }
            .readLines()
            .filter { it.isNotBlank() }
    }

    /**
     * Run YOLOv8n inference on [bitmap].
     * Executes on [Dispatchers.Default] (background thread).
     * Returns normalized [Detection] objects (coords in [0,1]) for the overlay.
     */
    suspend fun detect(bitmap: Bitmap): List<Detection> = withContext(Dispatchers.Default) {
        val inputBuffer = preprocessBitmap(bitmap)

        // Output buffer: [1, 84, 8400]
        val outputArray = Array(1) { Array(84) { FloatArray(NUM_ANCHORS) } }

        interpreter.run(inputBuffer, outputArray)

        val rawDetections = parseOutput(outputArray[0], bitmap.width, bitmap.height)
        applyNms(rawDetections)
    }

    // ── Pre-processing ────────────────────────────────────────────────────────

    /**
     * Resize bitmap to INPUT_SIZE × INPUT_SIZE and pack into a float32 ByteBuffer.
     * Pixel values normalized to [0, 1] in RGB channel order.
     */
    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Float32: 4 bytes × 3 channels × INPUT_SIZE²
        val buffer = ByteBuffer
            .allocateDirect(4 * 3 * INPUT_SIZE * INPUT_SIZE)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            buffer.putFloat(((pixel shr  8) and 0xFF) / 255.0f) // G
            buffer.putFloat(( pixel         and 0xFF) / 255.0f) // B
        }

        buffer.rewind()
        return buffer
    }

    // ── Output Parsing ────────────────────────────────────────────────────────

    /**
     * Converts raw [1, 84, 8400] model output into [Detection] objects.
     *
     * YOLOv8 TFLite output is transposed compared to ONNX:
     *   output[feature_index][anchor_index]
     *   - feature 0..3  → cx, cy, w, h (in INPUT_SIZE pixel space)
     *   - feature 4..83 → class_0 .. class_79 confidence scores
     *
     * Bounding boxes are converted to normalized [0,1] coords scaled to
     * the original camera frame dimensions for the overlay.
     */
    private fun parseOutput(
        output: Array<FloatArray>,
        origW: Int,
        origH: Int
    ): List<Detection> {
        val results = mutableListOf<Detection>()

        // Scale factors from INPUT_SIZE space back to original bitmap size
        val scaleX = origW.toFloat() / INPUT_SIZE
        val scaleY = origH.toFloat() / INPUT_SIZE

        for (i in 0 until NUM_ANCHORS) {
            // Center-format box in INPUT_SIZE pixel space
            val cx = output[0][i]
            val cy = output[1][i]
            val bw = output[2][i]
            val bh = output[3][i]

            // Find the highest-scoring class
            var maxScore = 0f
            var bestClass = 0
            for (c in 0 until NUM_CLASSES) {
                val score = output[4 + c][i]
                if (score > maxScore) {
                    maxScore = score
                    bestClass = c
                }
            }

            // Skip low-confidence predictions early
            if (maxScore < CONFIDENCE_THRESHOLD) continue

            // Convert center-format (cx, cy, w, h) to corner-format (left, top, right, bottom)
            // YOLOv8 TFLite output coordinates are in pixel space relative to INPUT_SIZE.
            // Normalize them to [0,1] before converting.
            val cxNorm = cx / INPUT_SIZE
            val cyNorm = cy / INPUT_SIZE
            val bwNorm = bw / INPUT_SIZE
            val bhNorm = bh / INPUT_SIZE
            val left   = cxNorm - bwNorm / 2f
            val top    = cyNorm - bhNorm / 2f
            val right  = cxNorm + bwNorm / 2f
            val bottom = cyNorm + bhNorm / 2f

            val box = RectF(
                left.coerceIn(0f, 1f),
                top.coerceIn(0f, 1f),
                right.coerceIn(0f, 1f),
                bottom.coerceIn(0f, 1f)
            )

            // Skip degenerate boxes (boxes that are too small to be meaningful)
            if (box.width() < 0.01f || box.height() < 0.01f) continue

            val label = if (bestClass < labels.size) labels[bestClass] else "object"
            results.add(Detection(label, maxScore, box))
        }

        return results
    }

    // ── Non-Maximum Suppression ───────────────────────────────────────────────

    /**
     * Greedy NMS: keep the highest-confidence box, remove overlapping boxes
     * that share class and have IoU > [IOU_THRESHOLD].
     */
    private fun applyNms(detections: List<Detection>): List<Detection> {
        // Group by class label to run NMS per-class
        val byClass = detections.groupBy { it.label }
        val kept    = mutableListOf<Detection>()

        for ((_, group) in byClass) {
            val sorted = group.sortedByDescending { it.confidence }.toMutableList()
            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                kept.add(best)
                sorted.removeAll { iou(best.boundingBox, it.boundingBox) > IOU_THRESHOLD }
            }
        }

        return kept
    }

    /** Computes Intersection-over-Union for two normalized bounding boxes. */
    private fun iou(a: RectF, b: RectF): Float {
        val interLeft   = maxOf(a.left,   b.left)
        val interTop    = maxOf(a.top,    b.top)
        val interRight  = minOf(a.right,  b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        val interW = (interRight  - interLeft).coerceAtLeast(0f)
        val interH = (interBottom - interTop ).coerceAtLeast(0f)
        val interArea = interW * interH

        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        val union = aArea + bArea - interArea

        return if (union <= 0f) 0f else interArea / union
    }
}
