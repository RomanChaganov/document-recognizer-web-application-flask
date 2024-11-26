from app import app
from app import WORK_DIR, TESSDATA_PATH
from app.preprocessing import preprocess
from app.tablepars import pars
from app.textrecognizer import recognize
from app.tablesrecognizer import recognize_generator

import cv2
from io import BytesIO

from flask import render_template
from flask import request
from flask import send_from_directory, send_file

from PIL import Image

from openpyxl import Workbook
import os
import tempfile
from werkzeug.utils import secure_filename

from tesserocr import PyTessBaseAPI, PSM, RIL
from tesserocr import iterate_level

from zipfile import ZipFile, ZIP_DEFLATED


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'), 
        'favicon.ico', 
        mimetype='image/vnd.microsoft.icon'
    )


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    delete_stamp = request.form.get('delete_stamp')
    
    if file and allowed_file(file.filename):
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as temp_dir:
            filename = secure_filename(file.filename)
            file.save(os.path.join(temp_dir, filename))
            
            processing(filename, temp_dir, delete_stamp == 'on')

            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zip_file.write(file_path, os.path.relpath(file_path, temp_dir))
            
            zip_buffer.seek(0)
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='archive.zip')

    return '<h2>Not OK</h2>'


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def processing(filename, temp_dir, delete_stamp):
    binary = preprocess(os.path.join(temp_dir, filename), delete_stamp)

    imgs, struct_sizes = pars(binary)
    workbook = Workbook()
    sheet = None

    with PyTessBaseAPI(path=TESSDATA_PATH, lang='rus+eng') as api:
        words = recognize(imgs[0], api)

        for i, cells_imgs in enumerate(recognize_generator(imgs[1], struct_sizes)):
            if i == 0:
                sheet = workbook.active
            else:
                sheet = workbook.create_sheet()
            
            sheet.title = f'Лист {i + 1}'

            _set_cells(cells_imgs, api, sheet)
        
    with open(os.path.join(temp_dir, 'text.txt'), 'w', encoding='UTF-8') as f:
        f.write(str(words))
    
    if sheet is not None:
        workbook.save(os.path.join(temp_dir, 'tables.xlsx'))
    
    cv2.imwrite(os.path.join(temp_dir, 'binary.jpg'), binary)


def _set_cells(cells_imgs, api, sheet):
    api.SetPageSegMode(PSM.SINGLE_BLOCK)

    for indexes, cell_img in cells_imgs:
        image = Image.fromarray(cell_img)
        api.SetImage(image)
        api.Recognize()

        iterator = api.GetIterator()
        level = RIL.WORD

        cell_text = []
        for itr in iterate_level(iterator, level):
            try:
                word = itr.GetUTF8Text(level)
            except RuntimeError:
                continue

            if not word.strip():
                continue

            cell_text.append(word)

        text = ' '.join(cell_text)

        sheet.cell(row=indexes[0] + 1, column=indexes[1] + 1, value=text)
