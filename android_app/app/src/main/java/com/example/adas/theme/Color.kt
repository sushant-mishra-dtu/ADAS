package com.example.adas.theme

import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

/**
 * Custom color palette for the ADAS system.
 * Exposes all semantic theme values for both Dark and Light modes.
 */
data class AdasColors(
    val background: Color,
    val surface: Color,
    val glass: Color,
    val glassStroke: Color,
    val cyan: Color,
    val pink: Color,
    val amber: Color,
    val green: Color,
    val purple: Color,
    val hudText: Color,
    val hudDim: Color,
    val textPrimary: Color,
    val danger: Color,
    val dangerDim: Color,
    val online: Color,
    val offline: Color
)

// ── Dark Theme Palette ───────────────────────────────────────────────────────
val DarkAdasColors = AdasColors(
    background   = Color(0xFF0A0A0F),
    surface      = Color(0xFF12121A),
    glass        = Color(0xCC0A0A0F),
    glassStroke  = Color(0x3300E5FF),
    cyan         = Color(0xFF00E5FF),
    pink         = Color(0xFFFF4081),
    amber        = Color(0xFFFFD740),
    green        = Color(0xFF69F0AE),
    purple       = Color(0xFFE040FB),
    hudText      = Color(0xFFB0BEC5),
    hudDim       = Color(0xFF546E7A),
    textPrimary  = Color(0xFFECEFF1),
    danger       = Color(0xFFFF1744),
    dangerDim    = Color(0x44FF1744),
    online       = Color(0xFF00E676),
    offline      = Color(0xFFFF5252)
)

// ── Light Theme Palette ──────────────────────────────────────────────────────
val LightAdasColors = AdasColors(
    background   = Color(0xFFF4F5F7),   // Light off-white
    surface      = Color(0xFFFFFFFF),   // Pure white cards
    glass        = Color(0xDDD9E1E8),   // Cool light gray glass
    glassStroke  = Color(0x4400838F),   // Muted cyan border
    cyan         = Color(0xFF00838F),   // Darker cyan for text readability
    pink         = Color(0xFFC2185B),   // Darker pink
    amber        = Color(0xFFF57C00),   // Darker amber
    green        = Color(0xFF2E7D32),   // Darker green
    purple       = Color(0xFF6A1B9A),   // Darker purple
    hudText      = Color(0xFF37474F),   // Charcoal text
    hudDim       = Color(0xFF78909C),   // Cool gray
    textPrimary  = Color(0xFF212121),   // Off-black text
    danger       = Color(0xFFD50000),   // Dark red alert
    dangerDim    = Color(0x33D50000),   // Faint red alert bg
    online       = Color(0xFF2E7D32),   // Dark green online dot
    offline      = Color(0xFFC62828)    // Offline red
)

val LocalAdasColors = staticCompositionLocalOf { DarkAdasColors }
