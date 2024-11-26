from flask import Flask

WORK_DIR = 'workdir'
TESSDATA_PATH = 'tessdata'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from app import preprocessing
from app import tablepars
from app import textrecognizer
from app import tablesrecognizer
from app import pipeline

from app import routes
