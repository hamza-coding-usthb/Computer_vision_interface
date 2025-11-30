# Object Detection, Calibration, and Position Calculation with Two Cameras

This project uses a stereo camera setup to detect an object and calculate its real-world 3D coordinates. The application provides a user interface built with Streamlit to manage the different steps of the process.

## Authors

This project was developed by:
- MIHOUBI Mohamed Anes
- TAOUCI Kenza
- TAOURIRT Hamza
- AITIDIR Abderahmane

## Project Overview

The core functionality of this project is to perform 3D reconstruction of an object's position using stereo vision. This is achieved through the following key steps:

1.  **Camera Calibration**: Determining the intrinsic and extrinsic parameters of the two cameras to understand their geometric relationship.
2.  **Object Detection**: Identifying and locating the target object in the video streams from both cameras.
3.  **Triangulation**: Using the calibration data and the detected object locations in both images to calculate the object's precise (X, Y, Z) coordinates in 3D space.

The entire process is wrapped in a user-friendly web interface powered by Streamlit.

## Getting Started

Follow these steps to set up and run the application.

### Prerequisites

- Python 3.7+
- `pip` package manager

### Installation and Execution

1.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install the required dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

3.  **Run the Streamlit application**:
    ```bash
    streamlit run part3_interface.py
    ```

Once the command is executed, the application should open in your default web browser.