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
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.adas.theme.AdasTheme
import com.example.adas.theme.HudFontFamily

/**
 * Glassmorphism HUD bar pinned to the bottom of the camera screen.
 * Contains 4 icon buttons: Record, Upload, Map/Log, Settings.
 * Fully supports dynamic Light/Dark theme configuration.
 */
@Composable
fun HudBottomBar(
    viewModel: AdasViewModel,
    onSettingsClick: () -> Unit,
    onLogClick: () -> Unit
) {
    val colors = AdasTheme.colors
    val isRecording by viewModel.isRecording.collectAsState()
    val uploadStatus by viewModel.uploadStatus.collectAsState()

    val infiniteTransition = rememberInfiniteTransition(label = "recPulse")
    val recAlpha by infiniteTransition.animateFloat(
        initialValue = 0.5f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "recAlpha"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .windowInsetsPadding(WindowInsets.navigationBars)
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(colors.glass)
                .padding(horizontal = 8.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // ── Record button ──────────────────────────────────────────────────
            HudButton(
                icon = "⏺",
                label = if (isRecording) "STOP" else "REC",
                tint = if (isRecording) colors.pink.copy(alpha = recAlpha) else colors.hudText,
                onClick = { viewModel.toggleRecording() }
            )

            // ── Upload button ──────────────────────────────────────────────────
            val uploadIcon = when (uploadStatus) {
                AdasViewModel.UploadStatus.UPLOADING -> "⏳"
                AdasViewModel.UploadStatus.SUCCESS   -> "✓"
                AdasViewModel.UploadStatus.FAILED    -> "✗"
                else                                 -> "▲"
            }
            HudButton(
                icon = uploadIcon,
                label = "UPLOAD",
                tint = when (uploadStatus) {
                    AdasViewModel.UploadStatus.UPLOADING -> colors.amber
                    AdasViewModel.UploadStatus.SUCCESS   -> colors.green
                    AdasViewModel.UploadStatus.FAILED    -> colors.offline
                    else                                 -> colors.hudText
                },
                onClick = { viewModel.triggerUpload() }
            )

            // ── Event Log button ───────────────────────────────────────────────
            HudButton(
                icon = "📋",
                label = "LOG",
                tint = colors.hudText,
                onClick = onLogClick
            )

            // ── Settings button ────────────────────────────────────────────────
            HudButton(
                icon = "⚙",
                label = "SET",
                tint = colors.hudText,
                onClick = onSettingsClick
            )
        }
    }
}

@Composable
private fun HudButton(
    icon: String,
    label: String,
    tint: Color,
    onClick: () -> Unit
) {
    val colors = AdasTheme.colors
    Column(
        modifier = Modifier
            .size(64.dp)
            .clip(RoundedCornerShape(10.dp))
            .clickable(role = Role.Button, onClick = onClick)
            .padding(6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = icon, fontSize = 22.sp, color = tint)
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = label,
            fontFamily = HudFontFamily,
            fontWeight = FontWeight.Medium,
            fontSize = 8.sp,
            color = colors.hudDim,
            letterSpacing = 0.8.sp
        )
    }
}
