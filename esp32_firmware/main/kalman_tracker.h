#pragma once

class KalmanTracker {
public:
    KalmanTracker();
    ~KalmanTracker();

    // Updates the tracker with a new bounding box measurement [x, y, w, h]
    void update(float x, float y, float w, float h);

    // Predicts the next state
    void predict();

    // Gets the current estimated bounding box
    void get_state(float& x, float& y, float& w, float& h);

private:
    // Simple 1D Kalman state variables for x, y, w, h for demonstration
    // In a real implementation, use a matrix library like Eigen or custom matrix math.
    float state[4];
    float uncertainty[4];
    float Q; // Process noise
    float R; // Measurement noise
};
