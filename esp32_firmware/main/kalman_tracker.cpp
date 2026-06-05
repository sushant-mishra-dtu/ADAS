#include "kalman_tracker.h"

KalmanTracker::KalmanTracker() {
    for (int i = 0; i < 4; ++i) {
        state[i] = 0.0f;
        uncertainty[i] = 1000.0f; // High initial uncertainty
    }
    Q = 0.1f;
    R = 1.0f;
}

KalmanTracker::~KalmanTracker() {}

void KalmanTracker::update(float x, float y, float w, float h) {
    float measurement[4] = {x, y, w, h};
    
    // Simplified 1D Kalman update for each variable
    for (int i = 0; i < 4; ++i) {
        // Kalman gain
        float K = uncertainty[i] / (uncertainty[i] + R);
        
        // Update estimate
        state[i] = state[i] + K * (measurement[i] - state[i]);
        
        // Update uncertainty
        uncertainty[i] = (1.0f - K) * uncertainty[i];
    }
}

void KalmanTracker::predict() {
    for (int i = 0; i < 4; ++i) {
        // Simple prediction (constant position model)
        // state[i] remains the same
        uncertainty[i] = uncertainty[i] + Q;
    }
}

void KalmanTracker::get_state(float& x, float& y, float& w, float& h) {
    x = state[0];
    y = state[1];
    w = state[2];
    h = state[3];
}
