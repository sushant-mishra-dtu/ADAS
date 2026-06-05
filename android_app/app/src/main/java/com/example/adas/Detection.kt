package com.example.adas

import android.graphics.RectF

/**
 * Represents a single detection result from the inference engine.
 *
 * @param label  The class label (e.g., "car", "person", "motorcycle").
 * @param confidence The confidence score from 0.0 to 1.0.
 * @param boundingBox The bounding box in normalized coordinates [0.0, 1.0]
 *                    relative to the image size (left, top, right, bottom).
 */
data class Detection(
    val label: String,
    val confidence: Float,
    val boundingBox: RectF
)
