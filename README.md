# Secundo-v-1.0

Secundo is an Optical Music Recognition (OMR) system that converts images of sheet music into MusicXML, a standard format supported by notation software such as MuseScore and Finale. The goal is to remove manual transcription by allowing users to upload scanned or photographed sheet music and receive an editable digital score.

The project is built in Python and combines classical image processing with a PyTorch-based CNN to recognize musical symbols. A preprocessing pipeline corrects skew, normalizes layout, and prepares note images for classification. Recognized symbols are then programmatically assembled into a valid MusicXML file.

Secundo includes a lightweight backend service and web interface that together provide an end-to-end pipeline from image upload to downloadable MusicXML output.

Key technologies: Python, PyTorch, OpenCV, CNNs, MusicXML, Flask
Focus areas: computer vision, deep learning, structured data generation, end-to-end ML systems
