package com.example.adas

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
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.adas.theme.AdasTheme
import com.example.adas.theme.HudFontFamily

/** Placeholder data class for a logged anomaly event */
data class AnomalyEvent(
    val id: Int,
    val timestamp: String,
    val label: String,
    val confidence: Float,
    val severity: String, // "HIGH" | "STANDARD"
    val location: String,
    val uploaded: Boolean
)

/**
 * Full-screen event log showing past detected anomaly clips.
 * Fully supports dynamic Light/Dark theme configuration.
 */
@Composable
fun EventLogScreen(onBack: () -> Unit) {
    val colors = AdasTheme.colors

    // Placeholder events (Phase D: replace with Room DB DAO calls)
    val events = remember {
        listOf(
            AnomalyEvent(1, "2026-06-05  22:14:03", "person",     0.88f, "HIGH",     "28.6139°N 77.2090°E", true),
            AnomalyEvent(2, "2026-06-05  21:57:41", "motorcycle", 0.74f, "STANDARD", "28.6145°N 77.2088°E", true),
            AnomalyEvent(3, "2026-06-05  21:30:12", "vehicle",    0.91f, "STANDARD", "28.6155°N 77.2070°E", false),
            AnomalyEvent(4, "2026-06-05  20:55:08", "person",     0.82f, "HIGH",     "28.6160°N 77.2065°E", false),
            AnomalyEvent(5, "2026-06-04  18:40:29", "truck",      0.67f, "STANDARD", "28.6120°N 77.2100°E", true),
            AnomalyEvent(6, "2026-06-04  17:22:55", "bus",        0.79f, "HIGH",     "28.6110°N 77.2110°E", true),
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {

            // ── Top bar ─────────────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .background(colors.glass)
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Text(text = "←", fontSize = 20.sp, color = colors.cyan)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text(
                        text = "ANOMALY EVENT LOG",
                        fontFamily = HudFontFamily,
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        letterSpacing = 2.sp,
                        color = colors.cyan
                    )
                    Text(
                        text = "${events.size} events · ${events.count { it.uploaded }} uploaded",
                        fontFamily = HudFontFamily,
                        fontSize = 10.sp,
                        color = colors.hudDim
                    )
                }
            }

            // ── Filter chips ─────────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FilterChip(label = "ALL",      selected = true)
                FilterChip(label = "🔴 HIGH",  selected = false)
                FilterChip(label = "✓ UPLOAD", selected = false)
                FilterChip(label = "⏳ PENDING",selected = false)
            }

            HorizontalDivider(color = colors.hudDim.copy(alpha = 0.2f))

            // ── Event list ───────────────────────────────────────────────────────
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                item { Spacer(modifier = Modifier.height(8.dp)) }
                items(events) { event ->
                    EventCard(event = event)
                }
                item { Spacer(modifier = Modifier.height(24.dp)) }
            }
        }
    }
}

@Composable
private fun EventCard(event: AnomalyEvent) {
    val colors = AdasTheme.colors
    val severityColor = if (event.severity == "HIGH") colors.pink else colors.amber
    val labelColor = when (event.label) {
        "person"     -> colors.pink
        "motorcycle" -> colors.amber
        "vehicle"    -> colors.cyan
        "truck"      -> colors.green
        "bus"        -> colors.purple
        else         -> colors.textPrimary
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(colors.surface)
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Severity indicator dot
        Box(
            modifier = Modifier
                .size(10.dp)
                .clip(CircleShape)
                .background(severityColor)
        )

        Spacer(modifier = Modifier.width(12.dp))

        Column(modifier = Modifier.weight(1f)) {
            // Class + confidence
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = event.label.uppercase(),
                    fontFamily = HudFontFamily,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                    color = labelColor
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "%.0f%%".format(event.confidence * 100),
                    fontFamily = HudFontFamily,
                    fontSize = 11.sp,
                    color = colors.hudText
                )
            }
            Spacer(modifier = Modifier.height(2.dp))
            // Timestamp
            Text(
                text = event.timestamp,
                fontFamily = HudFontFamily,
                fontSize = 10.sp,
                color = colors.hudDim
            )
            // GPS
            Text(
                text = "📍 ${event.location}",
                fontFamily = HudFontFamily,
                fontSize = 9.sp,
                color = colors.hudDim.copy(alpha = 0.7f)
            )
        }

        Spacer(modifier = Modifier.width(8.dp))

        Column(horizontalAlignment = Alignment.End) {
            Text(
                text = event.severity,
                fontFamily = HudFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = 9.sp,
                color = severityColor
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = if (event.uploaded) "✓ SYNCED" else "⏳ LOCAL",
                fontFamily = HudFontFamily,
                fontSize = 9.sp,
                color = if (event.uploaded) colors.green else colors.amber
            )
        }
    }
}

@Composable
private fun FilterChip(label: String, selected: Boolean) {
    val colors = AdasTheme.colors
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(
                if (selected) colors.cyan.copy(alpha = 0.15f)
                else colors.surface
            )
            .padding(horizontal = 12.dp, vertical = 6.dp)
    ) {
        Text(
            text = label,
            fontFamily = HudFontFamily,
            fontSize = 10.sp,
            color = if (selected) colors.cyan else colors.hudDim,
            letterSpacing = 0.8.sp
        )
    }
}
