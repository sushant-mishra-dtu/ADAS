package com.example.adas

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.adas.theme.AdasTheme
import com.example.adas.theme.HudFontFamily

/**
 * Full-screen camera permission rationale screen.
 * Fully supports dynamic Light/Dark theme configurations.
 */
@Composable
fun PermissionScreen(onRequest: () -> Unit) {
    val colors = AdasTheme.colors
    val infiniteTransition = rememberInfiniteTransition(label = "radar")

    // Radar sweep angle — rotates full 360° every 3 seconds
    val sweepAngle by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "sweep"
    )

    // Camera icon pulsing glow
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1200, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
    ) {
        // ── Radar Background ───────────────────────────────────────────────────
        Canvas(modifier = Modifier.fillMaxSize()) {
            val cx = size.width / 2f
            val cy = size.height * 0.35f
            val maxR = size.width * 0.45f

            // Concentric radar rings
            listOf(0.25f, 0.5f, 0.75f, 1.0f).forEach { fraction ->
                drawCircle(
                    color = colors.cyan.copy(alpha = 0.07f),
                    radius = maxR * fraction,
                    center = Offset(cx, cy),
                    style = Stroke(width = 1.dp.toPx())
                )
            }

            // Sweep arc (fading tail)
            drawArc(
                brush = Brush.sweepGradient(
                    colors = listOf(
                        Color.Transparent,
                        colors.cyan.copy(alpha = 0.0f),
                        colors.cyan.copy(alpha = 0.15f),
                        colors.cyan.copy(alpha = 0.35f),
                    ),
                    center = Offset(cx, cy)
                ),
                startAngle = sweepAngle - 60f,
                sweepAngle = 60f,
                useCenter = true,
                topLeft = Offset(cx - maxR, cy - maxR),
                size = androidx.compose.ui.geometry.Size(maxR * 2, maxR * 2),
            )

            // Sweep leading line
            val rad = Math.toRadians(sweepAngle.toDouble())
            drawLine(
                color = colors.cyan.copy(alpha = 0.8f),
                start = Offset(cx, cy),
                end = Offset(
                    cx + (maxR * kotlin.math.cos(rad)).toFloat(),
                    cy + (maxR * kotlin.math.sin(rad)).toFloat()
                ),
                strokeWidth = 1.5.dp.toPx(),
                cap = StrokeCap.Round
            )

            // Radar center dot
            drawCircle(
                color = colors.cyan.copy(alpha = 0.9f),
                radius = 4.dp.toPx(),
                center = Offset(cx, cy)
            )
        }

        // ── Content Column ─────────────────────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {

            Spacer(modifier = Modifier.height(24.dp))

            // Glowing camera icon ring
            Box(contentAlignment = Alignment.Center) {
                // Outer glow ring
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .clip(CircleShape)
                        .background(colors.cyan.copy(alpha = glowAlpha * 0.15f))
                )
                // Inner ring
                Box(
                    modifier = Modifier
                        .size(76.dp)
                        .clip(CircleShape)
                        .background(colors.cyan.copy(alpha = 0.1f))
                )
                // Camera emoji / icon
                Text(text = "📷", fontSize = 36.sp)
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Title
            Text(
                text = "ADAS CAMERA ACCESS",
                fontFamily = HudFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                letterSpacing = 2.sp,
                color = colors.cyan,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Required for real-time object detection",
                fontFamily = HudFontFamily,
                fontSize = 12.sp,
                color = colors.hudDim,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(36.dp))

            // Feature bullets
            PermissionBullet(icon = "🎯", text = "Real-time vehicle & pedestrian detection")
            Spacer(modifier = Modifier.height(12.dp))
            PermissionBullet(icon = "🔒", text = "On-device only — video never leaves your phone")
            Spacer(modifier = Modifier.height(12.dp))
            PermissionBullet(icon = "⚡", text = "NPU-accelerated inference at 20+ FPS")
            Spacer(modifier = Modifier.height(12.dp))
            PermissionBullet(icon = "🇮🇳", text = "Optimised for Indian road conditions")

            Spacer(modifier = Modifier.height(48.dp))

            // Gradient CTA Button
            Button(
                onClick = onRequest,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
                shape = RoundedCornerShape(8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.Transparent
                )
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            brush = Brush.horizontalGradient(
                                colors = listOf(colors.cyan, colors.purple)
                            ),
                            shape = RoundedCornerShape(8.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "GRANT CAMERA ACCESS",
                        fontFamily = HudFontFamily,
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        letterSpacing = 1.5.sp,
                        color = colors.background
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "Your camera feed is processed entirely on-device.",
                fontSize = 11.sp,
                color = colors.hudDim,
                textAlign = TextAlign.Center,
                fontFamily = HudFontFamily
            )
        }
    }
}

@Composable
private fun PermissionBullet(icon: String, text: String) {
    val colors = AdasTheme.colors
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(colors.cyan.copy(alpha = 0.1f)),
            contentAlignment = Alignment.Center
        ) {
            Text(text = icon, fontSize = 16.sp)
        }
        Spacer(modifier = Modifier.size(14.dp))
        Text(
            text = text,
            fontSize = 13.sp,
            color = colors.hudText,
            fontFamily = HudFontFamily
        )
    }
}
