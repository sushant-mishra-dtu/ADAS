package com.example.adas

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.adas.theme.AdasTheme
import com.example.adas.theme.HudFontFamily
import kotlinx.coroutines.delay

private data class BootStep(
    val label: String,
    val durationMs: Long,
    val colorHex: Long // Use hex values to represent color safely without referencing static Color instances
)

private val bootSequence = listOf(
    BootStep("INITIALIZING SENSORS",         600L,  0xFF00E5FF),
    BootStep("LOADING DETECTION MODEL",      800L,  0xFFFFD740),
    BootStep("BINDING CAMERAX PIPELINE",     500L,  0xFF00E5FF),
    BootStep("CALIBRATING CONFIDENCE GATE",  400L,  0xFFFFD740),
    BootStep("CAMERA SUBSYSTEM READY",       300L,  0xFF69F0AE),
)

/**
 * Splash/boot screen shown for ~2.6 seconds on cold start.
 * Fully updated to support dynamic colors from AdasTheme.
 */
@Composable
fun SplashScreen(onComplete: () -> Unit) {
    val colors = AdasTheme.colors
    val infiniteTransition = rememberInfiniteTransition(label = "scanline")

    // Scanline sweep Y position
    val scanlineY by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 2000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "scanlineY"
    )

    // Current boot step index
    var stepIndex by remember { mutableIntStateOf(0) }
    // Progress of the currently running step [0,1]
    var stepProgress by remember { mutableFloatStateOf(0f) }
    // Whether all steps are done
    var bootComplete by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        for (i in bootSequence.indices) {
            stepIndex = i
            val step = bootSequence[i]
            val tickMs = 16L
            val ticks = (step.durationMs / tickMs).coerceAtLeast(1)
            for (tick in 0..ticks) {
                stepProgress = tick.toFloat() / ticks
                delay(tickMs)
            }
        }
        bootComplete = true
        delay(400)
        onComplete()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
    ) {
        // ── Scanline overlay ────────────────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(2.dp)
                .align(Alignment.TopStart)
                .padding(top = (scanlineY * 900).dp)
                .background(colors.cyan.copy(alpha = 0.12f))
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.Start
        ) {

            // ── Logo block ─────────────────────────────────────────────────────
            Text(
                text = "█████╗ ██████╗  █████╗ ███████╗",
                fontFamily = HudFontFamily,
                fontSize = 8.sp,
                color = colors.cyan.copy(alpha = 0.5f),
                lineHeight = 10.sp
            )
            Text(
                text = "██╔══██╗██╔══██╗██╔══██╗██╔════╝",
                fontFamily = HudFontFamily,
                fontSize = 8.sp,
                color = colors.cyan.copy(alpha = 0.5f),
                lineHeight = 10.sp
            )
            Text(
                text = "███████║██║  ██║███████║███████╗",
                fontFamily = HudFontFamily,
                fontSize = 8.sp,
                color = colors.cyan.copy(alpha = 0.5f),
                lineHeight = 10.sp
            )
            Text(
                text = "██╔══██║██║  ██║██╔══██║╚════██║",
                fontFamily = HudFontFamily,
                fontSize = 8.sp,
                color = colors.cyan.copy(alpha = 0.5f),
                lineHeight = 10.sp
            )
            Text(
                text = "██║  ██║██████╔╝██║  ██║███████║",
                fontFamily = HudFontFamily,
                fontSize = 8.sp,
                color = colors.cyan.copy(alpha = 0.5f),
                lineHeight = 10.sp
            )

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "ADVANCED DRIVER ASSISTANCE SYSTEM",
                fontFamily = HudFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp,
                letterSpacing = 2.sp,
                color = colors.cyan
            )

            Text(
                text = "YOLOv8n · TFLite · India Roads · v1.0",
                fontFamily = HudFontFamily,
                fontSize = 10.sp,
                color = colors.hudDim,
                letterSpacing = 1.sp
            )

            Spacer(modifier = Modifier.height(48.dp))

            // ── Boot step rows ─────────────────────────────────────────────────
            bootSequence.forEachIndexed { index, step ->
                val isDone    = index < stepIndex || bootComplete
                val isActive  = index == stepIndex && !bootComplete

                BootStepRow(
                    label    = step.label,
                    progress = when {
                        isDone   -> 1f
                        isActive -> stepProgress
                        else     -> 0f
                    },
                    isDone   = isDone,
                    isActive = isActive,
                    color    = Color(step.colorHex)
                )
                Spacer(modifier = Modifier.height(14.dp))
            }

            Spacer(modifier = Modifier.height(32.dp))

            // ── System info footer ─────────────────────────────────────────────
            Text(
                text = "© 2026 ADAS PROJECT · MIT LICENSE · 🇮🇳 INDIA",
                fontFamily = HudFontFamily,
                fontSize = 9.sp,
                color = colors.hudDim.copy(alpha = 0.5f),
                letterSpacing = 1.sp
            )
        }
    }
}

@Composable
private fun BootStepRow(
    label: String,
    progress: Float,
    isDone: Boolean,
    isActive: Boolean,
    color: Color
) {
    val colors = AdasTheme.colors
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                fontFamily = HudFontFamily,
                fontSize = 11.sp,
                color = when {
                    isDone   -> colors.hudText
                    isActive -> color
                    else     -> colors.hudDim.copy(alpha = 0.4f)
                },
                letterSpacing = 0.8.sp,
                modifier = Modifier.weight(1f)
            )
            Spacer(modifier = Modifier.width(16.dp))
            Text(
                text = when {
                    isDone   -> "✓"
                    isActive -> "${(progress * 100).toInt()}%"
                    else     -> "..."
                },
                fontFamily = HudFontFamily,
                fontSize = 11.sp,
                color = when {
                    isDone   -> colors.green
                    isActive -> color
                    else     -> colors.hudDim.copy(alpha = 0.3f)
                }
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        LinearProgressIndicator(
            progress = { progress },
            modifier = Modifier.fillMaxWidth().height(2.dp),
            color = if (isDone) colors.green else color,
            trackColor = color.copy(alpha = 0.1f)
        )
    }
}
