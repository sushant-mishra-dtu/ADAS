package com.example.adas

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.adas.theme.ADASTheme
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ADASTheme {
                AdasApp()
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun AdasApp() {
    val cameraPermission = rememberPermissionState(android.Manifest.permission.CAMERA)

    when {
        cameraPermission.status.isGranted -> {
            // Camera permission granted — show the ADAS pipeline
            AdasCameraScreen(modifier = Modifier.fillMaxSize())
        }
        cameraPermission.status.shouldShowRationale -> {
            // Show rationale and a button to request again
            PermissionRationaleScreen(onRequest = { cameraPermission.launchPermissionRequest() })
        }
        else -> {
            // First launch — request immediately
            androidx.compose.runtime.LaunchedEffect(Unit) {
                cameraPermission.launchPermissionRequest()
            }
            PermissionRationaleScreen(onRequest = { cameraPermission.launchPermissionRequest() })
        }
    }
}

@Composable
fun PermissionRationaleScreen(onRequest: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        androidx.compose.foundation.layout.Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "📷 Camera Permission Required",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            androidx.compose.foundation.layout.Spacer(modifier = Modifier.padding(12.dp))
            Text(
                text = "The ADAS system needs camera access to detect vehicles, pedestrians, and road hazards in real-time.",
                fontSize = 14.sp,
                color = Color.LightGray
            )
            androidx.compose.foundation.layout.Spacer(modifier = Modifier.padding(24.dp))
            androidx.compose.material3.Button(onClick = onRequest) {
                Text("Grant Camera Access")
            }
        }
    }
}
