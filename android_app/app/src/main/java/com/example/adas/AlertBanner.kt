package com.example.adas

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.adas.theme.AdasTheme
import com.example.adas.theme.HudFontFamily

/**
 * Danger / caution banner that slides down from the top of the camera screen.
 * Appears when a pedestrian or motorcycle enters the center danger zone.
 * Pulses to draw the driver's attention.
 * Fully supports dynamic Light/Dark theme configurations.
 */
@Composable
fun AlertBanner(viewModel: AdasViewModel) {
    val colors = AdasTheme.colors
    val alertState by viewModel.alertState.collectAsState()

    val infiniteTransition = rememberInfiniteTransition(label = "alertPulse")
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.7f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(400, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    val isVisible = alertState.level != AdasViewModel.AlertLevel.NONE

    val bannerColor = when (alertState.level) {
        AdasViewModel.AlertLevel.DANGER  -> colors.danger.copy(alpha = pulseAlpha * 0.9f)
        AdasViewModel.AlertLevel.CAUTION -> colors.amber.copy(alpha = pulseAlpha * 0.85f)
        else                             -> colors.danger
    }

    AnimatedVisibility(
        visible = isVisible,
        enter = slideInVertically(initialOffsetY = { -it }),
        exit  = slideOutVertically(targetOffsetY  = { -it })
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .wrapContentHeight()
                .background(bannerColor)
                .padding(vertical = 10.dp, horizontal = 16.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = alertState.message,
                fontFamily = HudFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 13.sp,
                letterSpacing = 1.sp,
                color = Color.Black,
                textAlign = TextAlign.Center
            )
        }
    }
}
