package com.example.adas

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.adas.theme.AdasTheme
import com.example.adas.theme.HudFontFamily

/**
 * Glassmorphism HUD bar pinned to the top of the camera screen.
 * Fully supports dynamic Light/Dark theme configuration.
 */
@Composable
fun HudTopBar(
    viewModel: AdasViewModel,
    onBack: () -> Unit
) {
    val colors = AdasTheme.colors
    val fps by viewModel.fps.collectAsState()
    val objectCount by viewModel.objectCount.collectAsState()
    val uploadStatus by viewModel.uploadStatus.collectAsState()

    val infiniteTransition = rememberInfiniteTransition(label = "livePulse")
    val liveAlpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(800, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "liveAlpha"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .windowInsetsPadding(WindowInsets.statusBars)
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(44.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(colors.glass)
                .padding(horizontal = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // ── Left: Live status ──────────────────────────────────────────────
            Row(verticalAlignment = Alignment.CenterVertically) {
                // Back Button
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(4.dp))
                        .background(colors.cyan.copy(alpha = 0.12f))
                        .clickable(role = androidx.compose.ui.semantics.Role.Button, onClick = onBack)
                        .padding(horizontal = 8.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = "← BACK",
                        fontFamily = HudFontFamily,
                        fontWeight = FontWeight.Bold,
                        fontSize = 9.sp,
                        color = colors.cyan,
                        letterSpacing = 1.sp
                    )
                }
                Spacer(modifier = Modifier.width(10.dp))
                // Pulsing green dot
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(colors.green.copy(alpha = liveAlpha))
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "ADAS",
                    fontFamily = HudFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 11.sp,
                    letterSpacing = 1.5.sp,
                    color = colors.cyan
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "LIVE",
                    fontFamily = HudFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 11.sp,
                    letterSpacing = 1.5.sp,
                    color = colors.green
                )
            }

            // ── Center-left: FPS ──────────────────────────────────────────────
            HudMetric(label = "FPS", value = fps.toString())

            // ── Center: Object count ──────────────────────────────────────────
            HudMetric(
                label = "OBJ",
                value = objectCount.toString(),
                valueColor = if (objectCount > 0) colors.cyan else colors.hudDim
            )

            // ── Center-right: Upload status ───────────────────────────────────
            val uploadLabel = when (uploadStatus) {
                AdasViewModel.UploadStatus.UPLOADING -> "↑ SYNC"
                AdasViewModel.UploadStatus.SUCCESS   -> "✓ SYNC"
                AdasViewModel.UploadStatus.FAILED    -> "✗ SYNC"
                else                                 -> "CLOUD"
            }
            val uploadColor = when (uploadStatus) {
                AdasViewModel.UploadStatus.UPLOADING -> colors.cyan
                AdasViewModel.UploadStatus.SUCCESS   -> colors.green
                AdasViewModel.UploadStatus.FAILED    -> colors.offline
                else                                 -> colors.hudDim
            }
            Text(
                text = uploadLabel,
                fontFamily = HudFontFamily,
                fontWeight = FontWeight.Medium,
                fontSize = 10.sp,
                letterSpacing = 1.sp,
                color = uploadColor
            )

            // ── Right: Connection dot ─────────────────────────────────────────
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(7.dp)
                        .clip(CircleShape)
                        .background(colors.green)
                )
                Spacer(modifier = Modifier.width(5.dp))
                Text(
                    text = "WiFi",
                    fontFamily = HudFontFamily,
                    fontSize = 10.sp,
                    color = colors.hudText
                )
            }
        }
    }
}

@Composable
private fun HudMetric(
    label: String,
    value: String,
    valueColor: Color = AdasTheme.colors.textPrimary
) {
    val colors = AdasTheme.colors
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = label,
            fontFamily = HudFontFamily,
            fontSize = 9.sp,
            color = colors.hudDim,
            letterSpacing = 1.sp
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = value,
            fontFamily = HudFontFamily,
            fontWeight = FontWeight.Bold,
            fontSize = 13.sp,
            color = valueColor
        )
    }
}
