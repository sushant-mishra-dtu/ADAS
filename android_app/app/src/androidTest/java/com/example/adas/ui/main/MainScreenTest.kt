package com.example.adas.ui.main

import androidx.activity.ComponentActivity
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import com.example.adas.PermissionRationaleScreen
import org.junit.Before
import org.junit.Rule
import org.junit.Test

/** UI tests for the camera permission fallback screen. */
class MainScreenTest {

  @get:Rule val composeTestRule = createAndroidComposeRule<ComponentActivity>()

  @Before
  fun setup() {
    composeTestRule.setContent { PermissionRationaleScreen(onRequest = {}) }
  }

  @Test
  fun permissionPrompt_exists() {
    composeTestRule.onNodeWithText("Grant Camera Access").assertExists()
    composeTestRule.onNodeWithText(
      "The ADAS system needs camera access to detect vehicles, pedestrians, and road hazards in real-time."
    ).assertExists()
  }
}
