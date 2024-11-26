WORK_DIR = 'workdir'
TESSDATA_PATH = 'tessdata'

from flask import Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from app import init
from app import pipeline
from app import preprocessing
from app import tablepars
from app import textrecognizer
from app import tablesrecognizer

from app import routes
