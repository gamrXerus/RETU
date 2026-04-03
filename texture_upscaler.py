#!/usr/bin/env python3
# texture_upscaler.py - A Python Qt6 GUI for batch upscaling textures using realesrgan-ncnn-vulkan.

import os
import sys
import shutil
import subprocess
import shlex
from pathlib import Path
from typing import List
import traceback

# Qt6 imports - using PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QProgressBar,
    QFileDialog, QMessageBox, QLabel, QLineEdit, QGroupBox, QFormLayout, QHBoxLayout
)
from PySide6.QtCore import (
    Qt, QTimer, QObject, Signal, QThreadPool, QRunnable, Slot
)
from PySide6.QtGui import QIcon

# --- Constants ---
class Constants:
    """Stores constant values for the application."""
    DEFAULT_REALESRGAN_PATH = "./realesrgan-ncnn-vulkan"
    OUTPUT_FOLDER_NAME = "upscaled-textures"
    UI_UPDATE_INTERVAL_MS = 100

# --- Worker Thread for Background Processing ---
class Worker(QRunnable):
    """
    Worker thread to run tasks in the background, preventing the GUI from freezing.
    """
    class Signals(QObject):
        """Defines signals available from a running worker thread."""
        finished = Signal()
        error = Signal(str)
        progress = Signal(int)
        status = Signal(str)
        file_found = Signal(int)

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Worker.Signals()
        # Add the signals object to kwargs so the function can emit signals
        self.kwargs['signals'] = self.signals

    @Slot()
    def run(self):
        """Execute the worker's task."""
        try:
            self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

# --- Command Builder Utility ---
class CommandBuilder:
    """A helper class to safely build shell commands."""
    def __init__(self, executable_name: str):
        self.executable = executable_name
        self.arguments = []

    def add_flag(self, flag: str, value: str) -> 'CommandBuilder':
        """Adds a flag and its value (e.g., -i input.png)."""
        self.arguments.append(flag)
        self.arguments.append(value)
        return self

    def build_args_list(self) -> List[str]:
        """Builds the command as a list of arguments for subprocess."""
        return [self.executable] + self.arguments

# --- Main Application Window ---
class TextureUpscaler(QMainWindow):
    def __init__(self):
        super(TextureUpscaler, self).__init__()

        # --- Window Properties ---
        self.setWindowTitle("Texture Upscaler")
        self.setMinimumSize(500, 300)

        # --- Central Widget and Layout ---
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Application State ---
        self.input_folder = ""
        self.processing_canceled = False
        self.total_file_count = 0

        # --- Thread Pool for Background Tasks ---
        self.thread_pool = QThreadPool()
        print(f"Multithreading with maximum {self.thread_pool.maxThreadCount()} threads")

        # --- UI Components ---
        self.create_selection_group()
        self.create_settings_group()
        self.create_control_buttons()
        self.create_progress_section()
        self.layout.addStretch(1) # Pushes widgets to the top

    def create_selection_group(self):
        """Creates the UI group for selecting the input folder."""
        group = QGroupBox("Input", self)
        layout = QHBoxLayout(group)

        self.folder_path_label = QLabel("No folder selected", self)
        self.folder_path_label.setStyleSheet("font-style: italic;")

        self.select_folder_button = QPushButton("Select Texture Folder", self)
        self.select_folder_button.setIcon(QIcon.fromTheme("folder-open"))
        self.select_folder_button.clicked.connect(self.select_input_folder)

        layout.addWidget(self.folder_path_label, 1)
        layout.addWidget(self.select_folder_button)
        self.layout.addWidget(group)

    def create_settings_group(self):
        """Creates the UI group for application settings."""
        group = QGroupBox("Settings", self)
        layout = QFormLayout(group)

        # RealESRGAN path setting
        self.realesrgan_path_edit = QLineEdit(Constants.DEFAULT_REALESRGAN_PATH, self)
        browse_button = QPushButton("Browse", self)
        browse_button.clicked.connect(self.browse_realesrgan_path)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.realesrgan_path_edit)
        path_layout.addWidget(browse_button)
        layout.addRow("RealESRGAN Path:", path_layout)

        self.layout.addWidget(group)

    def create_control_buttons(self):
        """Creates the start and cancel buttons."""
        self.start_button = QPushButton("Start Upscaling (4x)", self)
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_button.clicked.connect(self.start_processing)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setIcon(QIcon.fromTheme("process-stop"))
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(button_layout)

    def create_progress_section(self):
        """Creates the UI section for status and progress."""
        self.status_label = QLabel("Ready", self)
        self.layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

    # --- Slots and Event Handlers ---

    def select_input_folder(self):
        """Opens a dialog to select the input directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Texture Folder")
        if folder:
            self.input_folder = folder
            self.folder_path_label.setText(folder)
            self.folder_path_label.setStyleSheet("") # Reset style
            self.status_label.setText("Ready to process.")
            self.progress_bar.setValue(0)

    def browse_realesrgan_path(self):
        """Opens a dialog to find the realesrgan executable."""
        path, _ = QFileDialog.getOpenFileName(self, "Select RealESRGAN Executable")
        if path:
            self.realesrgan_path_edit.setText(path)

    def start_processing(self):
        """Starts the texture upscaling process in a background thread."""
        if not self.input_folder:
            self.show_message("Please select an input folder first.", error=True)
            return

        realesrgan_path = self.realesrgan_path_edit.text()
        if not Path(realesrgan_path).is_file():
            self.show_message(
                f"RealESRGAN executable not found at:\n{realesrgan_path}",
                error=True
            )
            return

        self.set_ui_processing_state(True)
        self.processing_canceled = False
        self.progress_bar.setValue(0)
        self.total_file_count = 0

        # Create and start the worker thread
        worker = Worker(
            self.perform_texture_upscaling,
            input_dir=self.input_folder,
            realesrgan_path=realesrgan_path,
            model_name="realesrgan-x4plus", # General purpose model
            upscale_factor=4
        )

        worker.signals.status.connect(self.status_label.setText)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.file_found.connect(lambda count: setattr(self, 'total_file_count', count))
        worker.signals.finished.connect(self.on_processing_finished)
        worker.signals.error.connect(self.on_processing_error)

        self.thread_pool.start(worker)

    def cancel_processing(self):
        """Flags the processing to be canceled."""
        if self.thread_pool.activeThreadCount() > 0:
            self.status_label.setText("Canceling...")
            self.processing_canceled = True
            self.cancel_button.setEnabled(False)

    def on_processing_finished(self):
        """Called when the worker thread completes successfully."""
        if not self.processing_canceled:
            self.status_label.setText(f"Successfully upscaled {self.total_file_count} textures.")
            self.progress_bar.setValue(100)
            self.show_message("Upscaling process completed successfully!")
        else:
            self.status_label.setText("Processing canceled by user.")
            self.show_message("Processing was canceled.")
        self.set_ui_processing_state(False)

    def on_processing_error(self, error_message):
        """Called when the worker thread encounters an error."""
        self.status_label.setText("An error occurred.")
        self.show_message(f"An error occurred:\n{error_message}", error=True)
        self.set_ui_processing_state(False)

    # --- Core Logic ---

    def perform_texture_upscaling(self, input_dir, realesrgan_path, model_name, upscale_factor, signals):
        """
        The main function that finds and upscales all textures.
        This runs in the background worker thread.
        """
        input_path = Path(input_dir)
        output_path = input_path.parent / Constants.OUTPUT_FOLDER_NAME

        signals.status.emit("Searching for textures...")
        texture_files = list(input_path.rglob("*.png"))

        if not texture_files:
            signals.error.emit("No .png files found in the selected folder.")
            return

        total_files = len(texture_files)
        signals.file_found.emit(total_files)

        # Create the main output directory
        output_path.mkdir(exist_ok=True)

        for i, file_path in enumerate(texture_files):
            if self.processing_canceled:
                return

            # Update status for the current file
            progress_percent = int(((i + 1) / total_files) * 100)
            status_msg = f"[{i+1}/{total_files}] Upscaling: {file_path.name}"
            signals.status.emit(status_msg)
            signals.progress.emit(progress_percent)

            # Determine output path while preserving subfolder structure
            relative_path = file_path.relative_to(input_path)
            output_file_path = output_path / relative_path

            # Create subdirectories in the output folder if they don't exist
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Build and run the command
            cmd = CommandBuilder(realesrgan_path)
            cmd.add_flag("-i", str(file_path))
            cmd.add_flag("-o", str(output_file_path))
            cmd.add_flag("-n", model_name)
            cmd.add_flag("-s", str(upscale_factor))

            try:
                subprocess.run(
                    cmd.build_args_list(),
                    check=True,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
            except subprocess.CalledProcessError as e:
                error_details = f"Failed on file: {file_path.name}\n"
                error_details += f"Error: {e.stderr}"
                raise RuntimeError(error_details) from e

    # --- UI Helpers ---

    def set_ui_processing_state(self, is_processing):
        """Enables or disables UI elements based on processing state."""
        self.start_button.setEnabled(not is_processing)
        self.cancel_button.setEnabled(is_processing)
        self.realesrgan_path_edit.parent().parent().setEnabled(not is_processing)
        self.select_folder_button.setEnabled(not is_processing)


    def show_message(self, message, error=False):
        """Displays a message box to the user."""
        msg_box = QMessageBox(self)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Critical if error else QMessageBox.Information)
        msg_box.setWindowTitle("Error" if error else "Information")
        msg_box.exec()

    def closeEvent(self, event):
        """Ensures processing is canceled when the window is closed."""
        if self.thread_pool.activeThreadCount() > 0:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Processing is in progress. Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.cancel_processing()
                self.thread_pool.waitForDone(-1) # Wait for threads to finish
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# --- Main Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    try:
        window = TextureUpscaler()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(
            None, "Application Error",
            f"A critical error occurred on startup: {e}"
        )
        sys.exit(1)
