package com.example.adas.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// Monospace font family for HUD readouts (FPS, confidence %, timestamps)
// Uses system monospace as a reliable fallback — no asset needed
val HudFontFamily = FontFamily.Monospace

val AdasTypography = Typography(
    // App title / screen titles
    headlineLarge = TextStyle(
        fontFamily = HudFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 24.sp,
        lineHeight = 30.sp,
        letterSpacing = 2.sp
    ),
    // Section headers (HUD bar labels)
    headlineMedium = TextStyle(
        fontFamily = HudFontFamily,
        fontWeight = FontWeight.SemiBold,
        fontSize = 16.sp,
        lineHeight = 22.sp,
        letterSpacing = 1.5.sp
    ),
    // HUD live readouts (FPS, object count)
    titleLarge = TextStyle(
        fontFamily = HudFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 1.sp
    ),
    // Body / description text
    bodyLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.15.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.1.sp
    ),
    // Button labels
    labelLarge = TextStyle(
        fontFamily = HudFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 13.sp,
        lineHeight = 18.sp,
        letterSpacing = 1.sp
    ),
    // Small HUD tags / badge labels
    labelSmall = TextStyle(
        fontFamily = HudFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 10.sp,
        lineHeight = 14.sp,
        letterSpacing = 0.8.sp
    )
)
