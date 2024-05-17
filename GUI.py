# coding:utf-8

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import qtawesome
import torch
from model import AudioClassifier
from data_loader import AudioDataset
from torch.utils.data import DataLoader
import mysql.connector
from mysql.connector import Error
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pyaudio
import wave


class DepressionSurveyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout()

        questions = [
            "1. I feel depressed and down.",
            "*2. I feel best in the morning.",
            "3. I feel like crying or want to cry.",
            "4. I have trouble sleeping at night.",
            "*5. I eat as much as I used to.",
            "*6. My sexual function is normal.",
            "7. I feel I am losing weight.",
            "8. I am troubled by constipation.",
            "9. My heart beats faster than usual.",
            "10. I feel tired for no reason.",
            "*11. My mind is as clear as usual.",
            "*12. I do not find it difficult to do things as usual.",
            "13. I am restless and can't keep still.",
            "*14. I feel hopeful about the future.",
            "15. I get irritated more easily than usual.",
            "*16. I find it easy to make decisions.",
            "*17. I feel that I am a useful and necessary person.",
            "*18. My life is meaningful.",
            "19. If I died, others would be better off.",
            "*20. I still enjoy the things I usually enjoy."
        ]

        self.score_labels = []
        self.score_buttons = []
        for question in questions:
            label = QtWidgets.QLabel(question)
            self.score_labels.append(label)
            self.layout.addWidget(label)

            button_group = QtWidgets.QButtonGroup(self)
            score_layout = QtWidgets.QHBoxLayout()

            for i in range(1, 5):
                button = QtWidgets.QRadioButton(str(i))
                button_group.addButton(button, i)
                score_layout.addWidget(button)

            self.score_buttons.append(button_group)
            self.layout.addLayout(score_layout)

        self.submit_button = QtWidgets.QPushButton("Submit")
        self.submit_button.clicked.connect(self.calculate_score)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def calculate_score(self):
        score = 0
        reverse_scoring_items = [1, 4, 5, 10, 11, 13, 15, 16, 17, 19]  # Reverse scoring items

        for i, button_group in enumerate(self.score_buttons):
            if button_group.checkedId() != -1:
                value = button_group.checkedId()
                if i in reverse_scoring_items:
                    value = 5 - value  # Reverse scoring
                score += value

        # Calculate standard score
        standard_score = round(score * 1.25)

        # Interpret depression level based on standard score
        if standard_score <= 53:
            interpretation = "Normal"
        elif 53 < standard_score <= 62:
            interpretation = "Mild depression"
        elif 63 <= standard_score <= 72:
            interpretation = "Moderate depression"
        else:
            interpretation = "Severe depression"

        QtWidgets.QMessageBox.information(self, "Score", f"Your depression standard score is: {standard_score}, Assessment result: {interpretation}")


def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='en-US') 
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Service request error: {e}"


def connect_to_database(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection


def find_most_similar_question(connection, input_question):
    cursor = connection.cursor()
    query = "SELECT question FROM Depression Analysis"
    try:
        cursor.execute(query)
        questions = [row[0] for row in cursor.fetchall()]
        questions.append(input_question)
        
        vectorizer = TfidfVectorizer().fit_transform(questions)
        
        similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1])
        
        most_similar_index = similarities.argmax()
        return questions[most_similar_index]
    except Error as e:
        print(f"The error '{e}' occurred")

def get_answer_for_question(connection, question):
    cursor = connection.cursor()
    query = f"SELECT answer FROM Depression Analysis WHERE question = '{question}'"
    try:
        cursor.execute(query)
        answer = cursor.fetchone()
        return answer[0] if answer else "No answer found"
    except Error as e:
        print(f"The error '{e}' occurred")


def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")


# Load saved model
def load_model(model_path, device):
    model = AudioClassifier(input_size=13, hidden_size=64, num_layers=2, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Evaluation mode
    return model


def find_default_input_device_index():
    """
    Find and return the index of the default available input device.
    Returns None if not found.
    """
    p = pyaudio.PyAudio()
    default_device_index = None
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Check if the device can be used as input
            print(f"Found input device: {dev_info['name']}, Index: {i}")
            default_device_index = i
            break
    p.terminate()
    return default_device_index

    
# Predict WAV files
def predict_wav_files(model, file_paths, device):
    dataset = AudioDataset(file_paths, [0] * len(file_paths))  
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    with torch.no_grad():
        for features in loader:
            features = features[0].to(device)  
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
    return predictions


class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.prediction_label = QtWidgets.QLabel("Prediction Result")
        self.right_layout.addWidget(self.prediction_label, 6, 0, 1, 9)  

        self.answer_widget = QtWidgets.QTextEdit(self) 
        self.answer_widget.setReadOnly(True)  # Set to read-only
        self.right_layout.addWidget(self.answer_widget, 3, 0, 9, 10)  # Layout position

        self.left_button_1.clicked.connect(self.show_survey)
        self.left_button_2.clicked.connect(self.open_file_dialog)
        self.left_button_4.clicked.connect(self.match_sql_data)

    def record_audio(self, record_seconds=5, output_filename='output.wav'):
        """Record and save audio to file using the found default device"""
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 2
        fs = 44100
        
        default_device_index = find_default_input_device_index()
        if default_device_index is None:
            print("No valid input device found!")
            return
        
        p = pyaudio.PyAudio()
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True,
                        input_device_index=default_device_index)

        print(f"Start recording: Please speak within the next {record_seconds} seconds...")
        frames = []

        for i in range(0, int(fs / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print("Recording ended.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def match_sql_data(self):
        # File selection dialog
        host_name = "localhost"
        db_name = "part2"
        user_name = "root"
        user_password = "123456"

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav)")
        if file_name:
            # Convert audio file to WAV format
            wav_file_path = "temp.wav"
            convert_mp3_to_wav(file_name, wav_file_path)
            # Recognize question from audio
            question_text = speech_to_text(wav_file_path)
            print("Recognized question:", question_text)

            # Connect to database to find the most similar question and answer
            connection = connect_to_database(host_name, user_name, user_password, db_name)
            if connection is not None and question_text:
                similar_question = find_most_similar_question(connection, question_text)
                print("Most similar question:", similar_question)

                # Get the answer for the most similar question
                answer = get_answer_for_question(connection, similar_question)
                print("Answer:", answer)

                # Display matched answer in the interface
                self.answer_widget.clear()
                self.answer_widget.setText(f"Most similar question: {similar_question}\n\nAnswer: {answer}") 

    def open_file_dialog(self):
        # Open file selection
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Audio Files", "", "Audio Files (*.mp3 *.wav)")
        if file_names:
            # Use prediction function to predict the selected audio files

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = 'model_epoch_8.pth'  # Model file path
            model = load_model(model_path, device)

            predictions = predict_wav_files(model, file_names, device)
            print("Prediction results:", predictions)

            result_message = "Prediction results:\n" + "\n".join([f"{wav_file}: {pred}" for wav_file, pred in zip(file_names, predictions)])
            self.answer_widget.clear()

            # Construct result message and update the content of QTextEdit
            result_message = "Prediction results:\n" + "\n".join([f"{wav_file}: Prediction {pred}" for wav_file, pred in zip(file_names, predictions)])
            self.answer_widget.setText(result_message)

    def show_survey(self):
        self.survey_widget = DepressionSurveyWidget()
        self.survey_widget.show()

    def init_ui(self):
        self.setFixedSize(960, 700)
        self.main_widget = QtWidgets.QWidget()  # Create main window widget
        self.main_layout = QtWidgets.QGridLayout()  # Create grid layout for main widget
        self.main_widget.setLayout(self.main_layout)  # Set layout of main widget to grid layout

        self.left_widget = QtWidgets.QWidget()  # Create left side widget
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # Create grid layout for left widget
        self.left_widget.setLayout(self.left_layout)  # Set layout of left widget to grid

        self.right_widget = QtWidgets.QWidget()  # Create right side widget
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()  # Create grid layout for right widget
        self.right_widget.setLayout(self.right_layout)  # Set layout of right widget to grid

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 2)  # Left widget at row 0, column 0, spans 8 rows and 3 columns
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)  # Right widget at row 0, column 3, spans 8 rows and 9 columns
        self.setCentralWidget(self.main_widget)  # Set main widget as the central widget
        self.left_close = QtWidgets.QPushButton("")  # Close button
        self.left_visit = QtWidgets.QPushButton("")  # Blank button
        self.left_mini = QtWidgets.QPushButton("")  # Minimize button

        self.left_label_1 = QtWidgets.QPushButton("Part1 Table Audio Analysis")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QPushButton("Part2 Similarity Analysis")
        self.left_label_2.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("Contact and Help")
        self.left_label_3.setObjectName('left_label')

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "Table Depression Judgment")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.sellsy', color='white'), "Audio Prediction Analysis")
        self.left_button_2.setObjectName('left_button')
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.film', color='white'), "Record Audio")
        self.left_button_3.setObjectName('left_button')
        self.left_button_3.clicked.connect(lambda: self.record_audio(5, 'output.wav'))
        self.left_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.home', color='white'), "SQL Data Matching")
        self.left_button_4.setObjectName('left_button')
        self.left_button_5 = QtWidgets.QPushButton(qtawesome.icon('fa.download', color='white'), "None")
        self.left_button_5.setObjectName('left_button')
        self.left_button_6 = QtWidgets.QPushButton(qtawesome.icon('fa.heart', color='white'), "None")
        self.left_button_6.setObjectName('left_button')
        self.left_button_7 = QtWidgets.QPushButton(qtawesome.icon('fa.comment', color='white'), "Feedback and Suggestions")
        self.left_button_7.setObjectName('left_button')
        self.left_button_8 = QtWidgets.QPushButton(qtawesome.icon('fa.star', color='white'), "Follow")
        self.left_button_8.setObjectName('left_button')
        self.left_button_9 = QtWidgets.QPushButton(qtawesome.icon('fa.question', color='white'), "Encountered Issues")
        self.left_button_9.setObjectName('left_button')
        self.left_xxx = QtWidgets.QPushButton(" ")

        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_label_1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_2, 5, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_4, 6, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_5, 7, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_6, 8, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_3, 9, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_7, 10, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_8, 11, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_9, 12, 0, 1, 3)

        self.right_bar_widget = QtWidgets.QWidget()  # Right top search bar widget
        self.right_bar_layout = QtWidgets.QGridLayout()  # Right top search bar grid layout
        self.right_bar_widget.setLayout(self.right_bar_layout)
        self.search_icon = QtWidgets.QLabel(chr(0xf002) + ' ' + 'Search  ')
        self.search_icon.setFont(qtawesome.font('fa', 16))
        self.right_bar_widget_search_input = QtWidgets.QLineEdit()
        self.right_bar_widget_search_input.setPlaceholderText("Enter symptoms to search")

        self.right_bar_layout.addWidget(self.search_icon, 0, 0, 1, 1)
        self.right_bar_layout.addWidget(self.right_bar_widget_search_input, 0, 1, 1, 8)

        self.right_layout.addWidget(self.right_bar_widget, 0, 0, 1, 9)

        self.right_recommend_label = QtWidgets.QLabel("Depression Analysis Detection System")
        self.right_recommend_label.setObjectName('right_label')

        self.right_recommend_widget = QtWidgets.QWidget()  # Recommend cover widget
        self.right_recommend_layout = QtWidgets.QGridLayout()  # Recommend cover grid layout
        self.right_recommend_widget.setLayout(self.right_recommend_layout)

        self.recommend_button_1 = QtWidgets.QToolButton()
        self.recommend_button_1.setText("Depression Analysis")  # Set button text
        self.recommend_button_1.setIcon(QtGui.QIcon('./r1.png'))  # Set button icon
        self.recommend_button_1.setIconSize(QtCore.QSize(100, 100))  # Set icon size
        self.recommend_button_1.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # Set button style to icon above text

        self.recommend_button_2 = QtWidgets.QToolButton()
        self.recommend_button_2.setText("Depression Analysis")
        self.recommend_button_2.setIcon(QtGui.QIcon('./r2.png'))
        self.recommend_button_2.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_2.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.recommend_button_3 = QtWidgets.QToolButton()
        self.recommend_button_3.setText("Depression Analysis")
        self.recommend_button_3.setIcon(QtGui.QIcon('./r3.png'))
        self.recommend_button_3.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_3.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.recommend_button_4 = QtWidgets.QToolButton()
        self.recommend_button_4.setText("Depression Analysis")
        self.recommend_button_4.setIcon(QtGui.QIcon('./r4.png'))
        self.recommend_button_4.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_4.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.recommend_button_5 = QtWidgets.QToolButton()
        self.recommend_button_5.setText("Depression Analysis")
        self.recommend_button_5.setIcon(QtGui.QIcon('./r5.png'))
        self.recommend_button_5.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_5.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.right_recommend_layout.addWidget(self.recommend_button_1, 0, 0)
        self.right_recommend_layout.addWidget(self.recommend_button_2, 0, 1)
        self.right_recommend_layout.addWidget(self.recommend_button_3, 0, 2)
        self.right_recommend_layout.addWidget(self.recommend_button_4, 0, 3)
        self.right_recommend_layout.addWidget(self.recommend_button_5, 0, 4)

        self.right_layout.addWidget(self.right_recommend_label, 1, 0, 1, 9)
        self.right_layout.addWidget(self.right_recommend_widget, 2, 0, 2, 9)

        self.right_newsong_label = QtWidgets.QLabel("Depression Analysis")
        self.right_newsong_label.setObjectName('right_label')

        self.right_playlist_label = QtWidgets.QLabel("Depression Analysis")
        self.right_playlist_label.setObjectName('right_label')

        self.right_newsong_widget = QtWidgets.QWidget()  # New songs widget
        self.right_newsong_layout = QtWidgets.QGridLayout()  # New songs widget grid layout
        self.right_newsong_widget.setLayout(self.right_newsong_layout)

        self.newsong_button_1 = QtWidgets.QPushButton("Depression Analysis")
        self.newsong_button_2 = QtWidgets.QPushButton("Depression Analysis")
        self.newsong_button_3 = QtWidgets.QPushButton("Depression Analysis")
        self.newsong_button_4 = QtWidgets.QPushButton("Depression Analysis")
        self.newsong_button_5 = QtWidgets.QPushButton("Depression Analysis")
        self.newsong_button_6 = QtWidgets.QPushButton("Depression Analysis")
        self.right_newsong_layout.addWidget(self.newsong_button_1, 0, 1,)
        self.right_newsong_layout.addWidget(self.newsong_button_2, 1, 1,)
        self.right_newsong_layout.addWidget(self.newsong_button_3, 2, 1,)
        self.right_newsong_layout.addWidget(self.newsong_button_4, 3, 1,)
        self.right_newsong_layout.addWidget(self.newsong_button_5, 4, 1,)
        self.right_newsong_layout.addWidget(self.newsong_button_6, 5, 1,)

        self.right_playlist_widget = QtWidgets.QWidget()  # Playlist widget
        self.right_playlist_layout = QtWidgets.QGridLayout()  # Playlist widget grid layout
        self.right_playlist_widget.setLayout(self.right_playlist_layout)

        self.playlist_button_1 = QtWidgets.QToolButton()
        self.playlist_button_1.setText("Depression Analysis")
        self.playlist_button_1.setIcon(QtGui.QIcon('./p1.jpg'))
        self.playlist_button_1.setIconSize(QtCore.QSize(100, 100))
        self.playlist_button_1.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.playlist_button_2 = QtWidgets.QToolButton()
        self.playlist_button_2.setText("Depression Analysis")
        self.playlist_button_2.setIcon(QtGui.QIcon('./p2.jpg'))
        self.playlist_button_2.setIconSize(QtCore.QSize(100, 100))
        self.playlist_button_2.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.playlist_button_3 = QtWidgets.QToolButton()
        self.playlist_button_3.setText("Depression Analysis")
        self.playlist_button_3.setIcon(QtGui.QIcon('./p3.jpg'))
        self.playlist_button_3.setIconSize(QtCore.QSize(100, 100))
        self.playlist_button_3.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.playlist_button_4 = QtWidgets.QToolButton()
        self.playlist_button_4.setText("Depression Analysis")
        self.playlist_button_4.setIcon(QtGui.QIcon('./p4.jpg'))
        self.playlist_button_4.setIconSize(QtCore.QSize(100, 100))
        self.playlist_button_4.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.right_playlist_layout.addWidget(self.playlist_button_1, 0, 0)
        self.right_playlist_layout.addWidget(self.playlist_button_2, 0, 1)
        self.right_playlist_layout.addWidget(self.playlist_button_3, 1, 0)
        self.right_playlist_layout.addWidget(self.playlist_button_4, 1, 1)

        self.right_layout.addWidget(self.right_newsong_label, 4, 0, 1, 5)
        self.right_layout.addWidget(self.right_playlist_label, 4, 5, 1, 4)
        self.right_layout.addWidget(self.right_newsong_widget, 5, 0, 1, 5)
        self.right_layout.addWidget(self.right_playlist_widget, 5, 5, 1, 4)


        self.right_process_bar = QtWidgets.QProgressBar()  # Playback progress bar widget
        self.right_process_bar.setValue(49)
        self.right_process_bar.setFixedHeight(3)  # Set progress bar height
        self.right_process_bar.setTextVisible(False)  # Do not display progress bar text

        self.right_playconsole_widget = QtWidgets.QWidget()  # Playback control widget
        self.right_playconsole_layout = QtWidgets.QGridLayout()  # Playback control widget grid layout
        self.right_playconsole_widget.setLayout(self.right_playconsole_layout)

        self.console_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.backward', color='#F76677'), "")
        self.console_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.forward', color='#F76677'), "")
        self.console_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.pause', color='#F76677', font=18), "")
        self.console_button_3.setIconSize(QtCore.QSize(30, 30))

        self.right_playconsole_layout.addWidget(self.console_button_1, 0, 0)
        self.right_playconsole_layout.addWidget(self.console_button_2, 0, 2)
        self.right_playconsole_layout.addWidget(self.console_button_3, 0, 1)
        self.right_playconsole_layout.setAlignment(QtCore.Qt.AlignCenter)  # Set layout components to center

        self.right_layout.addWidget(self.right_process_bar, 9, 0, 1, 9)
        self.right_layout.addWidget(self.right_playconsole_widget, 10, 0, 1, 9)


        self.left_close.setFixedSize(15, 15)  # Set close button size
        self.left_visit.setFixedSize(15, 15)  # Set button size
        self.left_mini.setFixedSize(15, 15)  # Set minimize button size


        self.left_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet('''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')


        self.left_widget.setStyleSheet("\
            QPushButton{border:none;color:white;}\
            QPushButton#left_label{\
                border:none;\
                border-bottom:1px solid white;\
                font-size:18px;\
                font-weight:700;\
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;\
            }\
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}\
            QWidget#left_widget{\
                background:gray;\
                border-top:1px solid white;\
                border-bottom:1px solid white;\
                border-left:1px solid white;\
                border-top-left-radius:10px;\
                border-bottom-left-radius:10px;\
            }"
        )


        self.right_bar_widget_search_input.setStyleSheet(
        "QLineEdit{\
                border:1px solid gray;\
                width:300px;\
                border-radius:10px;\
                padding:2px 4px;\
        }")


        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_label{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')


        self.right_recommend_widget.setStyleSheet(
            '''
                QToolButton{border:none;}
                QToolButton:hover{border-bottom:2px solid #F76677;}
            ''')
        self.right_playlist_widget.setStyleSheet(
            '''
                QToolButton{border:none;}
                QToolButton:hover{border-bottom:2px solid #F76677;}
            ''')


        self.right_newsong_widget.setStyleSheet('''
            QPushButton{
                border:none;
                color:gray;
                font-size:12px;
                height:40px;
                padding-left:5px;
                padding-right:10px;
                text-align:left;
            }
            QPushButton:hover{
                color:black;
                border:1px solid #F3F3F5;
                border-radius:10px;
                background:LightGray;
            }
        ''')


        self.right_process_bar.setStyleSheet('''
            QProgressBar::chunk {
                background-color: #F76677;
            }
        ''')

        self.right_playconsole_widget.setStyleSheet('''
            QPushButton{
                border:none;
            }
        ''')


        self.setWindowOpacity(0.9)  # Set window opacity
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # Set window background transparent


        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # Hide borders

        self.main_layout.setSpacing(0)


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
