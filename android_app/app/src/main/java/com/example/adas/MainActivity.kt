package com.example.adas

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.example.adas.theme.ADASTheme
import com.example.adas.theme.AdasTheme
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState

enum class AdasScreen {
    HOME,
    CAMERA,
    EVENT_LOG,
    PERMISSION_REQUEST
}

class MainActivity : ComponentActivity() {

    private val viewModel: AdasViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Keep screen on while the ADAS system is active
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            val isDarkTheme by viewModel.isDarkTheme.collectAsState()
            ADASTheme(isDarkTheme = isDarkTheme) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(AdasTheme.colors.background)
                ) {
                    AdasApp(viewModel = viewModel)
                }
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun AdasApp(viewModel: AdasViewModel) {
    var currentScreen by remember { mutableStateOf(AdasScreen.HOME) }
    var showSettingsSheet by remember { mutableStateOf(false) }

    val cameraPermission = rememberPermissionState(android.Manifest.permission.CAMERA)

    // Automatically transition to CAMERA if permission is granted while on the PERMISSION_REQUEST screen
    LaunchedEffect(cameraPermission.status.isGranted) {
        if (cameraPermission.status.isGranted && currentScreen == AdasScreen.PERMISSION_REQUEST) {
            currentScreen = AdasScreen.CAMERA
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        when (currentScreen) {
            AdasScreen.HOME -> {
                HomeScreen(
                    onStartADAS = {
                        if (cameraPermission.status.isGranted) {
                            currentScreen = AdasScreen.CAMERA
                        } else {
                            currentScreen = AdasScreen.PERMISSION_REQUEST
                        }
                    },
                    onOpenLog = {
                        currentScreen = AdasScreen.EVENT_LOG
                    },
                    onOpenSettings = {
                        showSettingsSheet = true
                    },
                    viewModel = viewModel
                )
            }
            AdasScreen.CAMERA -> {
                AdasCameraScreen(
                    viewModel = viewModel,
                    onBack = {
                        currentScreen = AdasScreen.HOME
                    }
                )
            }
            AdasScreen.EVENT_LOG -> {
                EventLogScreen(
                    onBack = {
                        currentScreen = AdasScreen.HOME
                    }
                )
            }
            AdasScreen.PERMISSION_REQUEST -> {
                PermissionScreen(
                    onRequest = {
                        cameraPermission.launchPermissionRequest()
                    }
                )
            }
        }

        if (showSettingsSheet) {
            SettingsSheet(
                viewModel = viewModel,
                onDismiss = { showSettingsSheet = false }
            )
        }
    }
}
