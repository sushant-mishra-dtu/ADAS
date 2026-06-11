package com.example.adas.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider

/**
 * Convenience theme object to retrieve the custom ADAS colors.
 * Usage: AdasTheme.colors.background
 */
object AdasTheme {
    val colors: AdasColors
        @Composable
        get() = LocalAdasColors.current
}

@Composable
fun ADASTheme(
    isDarkTheme: Boolean = true,
    content: @Composable () -> Unit
) {
    val adasColors = if (isDarkTheme) DarkAdasColors else LightAdasColors

    val materialColorScheme = if (isDarkTheme) {
        darkColorScheme(
            primary      = adasColors.cyan,
            onPrimary    = adasColors.background,
            secondary    = adasColors.purple,
            onSecondary  = adasColors.background,
            background   = adasColors.background,
            onBackground = adasColors.textPrimary,
            surface      = adasColors.surface,
            onSurface    = adasColors.textPrimary,
            error        = adasColors.danger,
            onError      = adasColors.textPrimary
        )
    } else {
        lightColorScheme(
            primary      = adasColors.cyan,
            onPrimary    = adasColors.surface,
            secondary    = adasColors.purple,
            onSecondary  = adasColors.surface,
            background   = adasColors.background,
            onBackground = adasColors.textPrimary,
            surface      = adasColors.surface,
            onSurface    = adasColors.textPrimary,
            error        = adasColors.danger,
            onError      = adasColors.surface
        )
    }

    CompositionLocalProvider(LocalAdasColors provides adasColors) {
        MaterialTheme(
            colorScheme = materialColorScheme,
            typography  = AdasTypography,
            content     = content
        )
    }
}
