import sys
import pandas as pd
import pickle
from ydata_profiling import ProfileReport
from pycaret.classification import setup,pull,compare_models,save_model
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QTableWidget, QAbstractItemView, QComboBox, QProgressBar, QTabWidget, QTabBar, QScrollArea
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Sanchayan's AutoML App (Prototype)[Classification Problems Only]")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget(self)
        self.layout.addWidget(self.tabs)

        self.upload_tab = QWidget(self)
        self.tabs.addTab(self.upload_tab, "Upload")

        self.upload_layout = QVBoxLayout(self.upload_tab)

        self.upload_button = QPushButton("Upload Dataset", self.upload_tab)
        self.upload_layout.addWidget(self.upload_button)

        self.upload_button.clicked.connect(self.upload_file)

        self.eda_tab = QWidget(self)
        self.tabs.addTab(self.eda_tab, "EDA")

        self.eda_layout = QVBoxLayout(self.eda_tab)

        self.eda_report = QLabel(self.eda_tab)
        self.eda_layout.addWidget(self.eda_report)

        self.ml_tab = QWidget(self)
        self.tabs.addTab(self.ml_tab, "ML")

        self.ml_layout = QVBoxLayout(self.ml_tab)

        self.target_label = QLabel("Select Your Target", self.ml_tab)
        self.ml_layout.addWidget(self.target_label)

        self.target_combo = QComboBox(self.ml_tab)
        self.ml_layout.addWidget(self.target_combo)

        self.train_button = QPushButton("Train Model", self.ml_tab)
        self.ml_layout.addWidget(self.train_button)

        self.train_button.clicked.connect(self.train_model)

        self.pipeline_tab = QWidget(self)
        self.tabs.addTab(self.pipeline_tab, "Pipeline Download")

        self.pipeline_layout = QVBoxLayout(self.pipeline_tab)

        self.download_button = QPushButton("Download the file", self.pipeline_tab)
        self.pipeline_layout.addWidget(self.download_button)

        self.download_button.clicked.connect(self.download_model)

        self.show()

    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv)")
        if file_name:
            self.df = pd.read_csv(file_name)
            self.upload_button.setText("Dataset Uploaded")
            self.upload_button.setEnabled(False)

    def train_model(self):
        if self.df is not None:
            target = self.target_combo.currentText()
            setup(data=self.df, target=target, verbose=False, use_gpu=False)
            setup_df = pull()
            self.eda_report.setText(str(setup_df))
            best_model = compare_models()
            compare_df = pull()
            self.eda_report.setText(str(compare_df))
            best_model
            save_model(best_model, "best_model.pkl")

    def download_model(self):
        with open("best_model.pkl", "rb") as f:
            model_data = f.read()
        model_mimetype = "application/octet-stream"
        model_name = "trained_model.pkl"
        model_qbytearray = QByteArray(model_data)
        model_qmimedata = QMimeData()
        model_qmimedata.setData(model_mimetype, model_qbytearray)
        model_qdrag = QDrag(self)
        model_qdrag.setMimeData(model_qmimedata)
        model_qdrag.exec_(Qt.CopyAction)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())