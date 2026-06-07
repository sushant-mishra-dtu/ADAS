package com.example.adas

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors

/**
 * Full-screen camera preview composable with real-time ADAS detection overlay.
 *
 * Binds two CameraX use-cases to the host lifecycle:
 *  1. [Preview]       → drives the [PreviewView] (full-screen camera feed)
 *  2. [ImageAnalysis] → feeds frames to [FrameAnalyzer] → [InferenceEngine]
 *
 * Detection results from the inference engine are reflected immediately via
 * [DetectionOverlay], which draws bounding boxes on a transparent Compose Canvas.
 */
@Composable
fun AdasCameraScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val inferenceEngine = remember { InferenceEngine(context) }
    var detections by remember { mutableStateOf(emptyList<Detection>()) }
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    val analyzer = remember(inferenceEngine) {
        FrameAnalyzer(inferenceEngine) { results ->
            detections = results
        }
    }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    DisposableEffect(cameraProviderFuture, analyzer, analysisExecutor) {
        onDispose {
            analyzer.close()
            analysisExecutor.shutdown()
            if (cameraProviderFuture.isDone) {
                cameraProviderFuture.get().unbindAll()
            }
        }
    }

    // Bind CameraX use-cases once
    LaunchedEffect(cameraProviderFuture, lifecycleOwner, previewView, analyzer, analysisExecutor) {
        val cameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        androidx.camera.core.resolutionselector.ResolutionStrategy(
                            android.util.Size(640, 480),
                            androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { it.setAnalyzer(analysisExecutor, analyzer) }

        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )
    }

    Box(modifier = modifier.fillMaxSize()) {
        // Layer 1: Full-screen camera feed (Android View)
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        // Layer 2: Transparent detection overlay (Compose Canvas)
        DetectionOverlay(
            detections = detections,
            modifier = Modifier.fillMaxSize()
        )
    }
}
