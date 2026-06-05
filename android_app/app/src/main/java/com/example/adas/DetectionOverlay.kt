package com.example.adas

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Color scheme for each detected class label */
private val labelColors = mapOf(
    "vehicle"    to Color(0xFF00E5FF),   // Cyan
    "person"     to Color(0xFFFF4081),   // Pink
    "motorcycle" to Color(0xFFFFD740),   // Amber
    "truck"      to Color(0xFF69F0AE),   // Green
    "bus"        to Color(0xFFE040FB),   // Purple
)
private val defaultColor = Color(0xFFFFFFFF) // White fallback

/**
 * Transparent Compose [Canvas] overlay that draws bounding boxes and labels
 * for each [Detection] on top of the CameraX preview.
 *
 * Coordinates in [detections] are assumed to be in normalized [0, 1] space
 * relative to the image dimensions. This composable scales them to screen pixels.
 */
@Composable
fun DetectionOverlay(
    detections: List<Detection>,
    modifier: Modifier = Modifier
) {
    val textMeasurer = rememberTextMeasurer()

    Canvas(modifier = modifier.fillMaxSize()) {
        detections.forEach { detection ->
            drawDetection(detection, textMeasurer)
        }
    }
}

private fun DrawScope.drawDetection(detection: Detection, textMeasurer: TextMeasurer) {
    val box = detection.boundingBox
    val color = labelColors[detection.label] ?: defaultColor

    val left   = box.left   * size.width
    val top    = box.top    * size.height
    val right  = box.right  * size.width
    val bottom = box.bottom * size.height
    val w = right - left
    val h = bottom - top

    // ── Bounding box ─────────────────────────────────────────────────────────
    drawRect(
        color = color,
        topLeft = Offset(left, top),
        size = Size(w, h),
        style = Stroke(width = 2.5.dp.toPx())
    )

    // ── Corner accent marks ────────────────────────────────────────────────
    val cornerLen = minOf(w, h) * 0.18f
    val strokePx = 3.5.dp.toPx()
    val corners = listOf(
        // top-left
        Pair(Offset(left, top + cornerLen), Offset(left, top)) to Offset(left + cornerLen, top),
        // top-right
        Pair(Offset(right - cornerLen, top), Offset(right, top)) to Offset(right, top + cornerLen),
        // bottom-left
        Pair(Offset(left, bottom - cornerLen), Offset(left, bottom)) to Offset(left + cornerLen, bottom),
        // bottom-right
        Pair(Offset(right - cornerLen, bottom), Offset(right, bottom)) to Offset(right, bottom - cornerLen),
    )
    corners.forEach { (vPair, hEnd) ->
        drawLine(color, vPair.first, vPair.second, strokeWidth = strokePx)
        drawLine(color, vPair.second, hEnd, strokeWidth = strokePx)
    }

    // ── Label badge ────────────────────────────────────────────────────────
    val labelText = "${detection.label}  ${"%.0f".format(detection.confidence * 100)}%"
    val textStyle = TextStyle(
        color = Color.Black,
        fontSize = 11.sp,
        fontWeight = FontWeight.SemiBold,
    )
    val measured = textMeasurer.measure(labelText, textStyle)
    val badgePad = 4.dp.toPx()
    val badgeW = measured.size.width + badgePad * 2
    val badgeH = measured.size.height + badgePad * 2
    val badgeTop = (top - badgeH).coerceAtLeast(0f)

    drawRect(
        color = color,
        topLeft = Offset(left, badgeTop),
        size = Size(badgeW, badgeH)
    )
    drawText(
        textMeasurer = textMeasurer,
        text = labelText,
        style = textStyle,
        topLeft = Offset(left + badgePad, badgeTop + badgePad)
    )
}
