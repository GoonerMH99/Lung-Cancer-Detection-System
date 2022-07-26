from flask import Flask,render_template,request,redirect #to get the output
from flask_mysqldb import MySQL
from PIL import Image
import numpy as np
import pydicom
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing
from skimage.morphology import reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from scipy import ndimage
import scipy.misc
import skimage
from tensorflow.keras.models import load_model
import cv2
from tempfile import TemporaryFile
from werkzeug.utils import secure_filename
def get_segmented_lungs(im, plot=False):
    """
    This function segments the lungs from the given 2D slice.
    """
    '''Step 1: Convert into a binary image. '''
    binary = im < 604
    '''Step 2: Remove the blobs connected to the border of the image.'''
    cleared = skimage.segmentation.clear_border(binary)
    ''' Step 3: Label the image.'''
    label_image = label(cleared)
    '''Step 4: Keep the labels with 2 largest areas.'''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    '''Step 5: Erosion operation with a disk of radius 2. This operation is 
    separate the lung nodules attached to the blood vessels.'''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    '''Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.'''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    '''Step 7: Fill in the small holes inside the binary mask of lungs.'''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    '''Step 8: Superimpose the binary mask on the input image.'''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    return im

#app= Flask(__name__)
#app.config['IMAGES UPLOADS']='C:/Users/ali mohamed/Desktop/pythonProject1/process imgs/static/images'#folder where the photo will be saved
app=Flask(__name__)
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='patients'
mysql=MySQL(app)
@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/info',methods=['GET','POST'])
def info():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        date = request.form['date']
        textarea = request.form['message']
        smokingH= request.form.get('selectS')
        gentic=request.form.get('selectG')
        Diabetes=request.form.get('selectD')
        Heartdisease=request.form.get('selectH')
        BloodPressure=request.form.get('selectB')
        cur = mysql.connection.cursor()
        record = (name, email,gentic,smokingH,BloodPressure,Heartdisease,Diabetes,date,textarea)
        ssqql = '''INSERT INTO info (Name,Email,genticCancer,SmokingHistory,BloodPressure,Heartdisease,Diabetes,Age,AdditionalDetails) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
        cur.execute(ssqql,record)
        mysql.connection.commit()
        cur.close()
        return render_template('img.html')
    return render_template('info.html')
@app.route('/Related Articles')
def RelatedArticles():
    return render_template('news.html')
@app.route('/about')
def about():
    return render_template('aboutUs.html')
@app.route('/contact',methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['Phone']
        age = request.form['Age']
        textarea = request.form['message']
        cur = mysql.connection.cursor()
        record = (name, email, phone,age, textarea)
        ssqql = '''INSERT INTO contact (Name,Email,Phone,Age,message) VALUES(%s,%s,%s,%s,%s)'''
        cur.execute(ssqql, record)
        mysql.connection.commit()
        cur.close()
        return render_template('contactoutput.html')
    return render_template('contactus.html')

@app.route("/upload",methods=["POST","GET"])
def upload_img():
    if request.method == "POST":
        image = request.files['file']
        if image.filename == '':
            print("Image must have a file name")
            return redirect(request.url)
        filename = image.filename
      #  img = Image.open(image)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):  # img = Image.open(image)
                img = Image.open(image)
                data = np.asarray(img, dtype="int32")
                pp=get_segmented_lungs(data)
                pp = cv2.resize(pp, (50, 50))
                X_test = pp.reshape(1, 50, 50, 1)
                model = load_model('model(2Layers-2nodesOP-93acc).h5')
                pred = model.predict(X_test)
                print(pred)
                cancer = "Cancer"
                normal = "Normal"
                if pred[0][1] > 0.5:
                    cur = mysql.connection.cursor()
                    cur.execute("UPDATE info SET prediction = 'cancer' WHERE id = ( SELECT MAX(id) FROM info )")
                    mysql.connection.commit()
                    cur.close()
                    return render_template('img.html', cancer=cancer)
                else:
                    cur = mysql.connection.cursor()
                    cur.execute("UPDATE info SET prediction = 'normal' WHERE id = ( SELECT MAX(id) FROM info )")
                    mysql.connection.commit()
                    cur.close()
                    return  render_template('img.html',normal=normal)

        else:
              dicom = pydicom.dcmread(image)
              np_pixel_array = dicom.pixel_array
              pp=get_segmented_lungs(np_pixel_array)
              pp = cv2.resize(pp, (50, 50))
              X_test = pp.reshape(1, 50, 50, 1)
              model = load_model('model(2Layers-2nodesOP-93acc).h5')
              pred = model.predict(X_test)
              print(pred)
              cancer="Cancer"
              normal="normal"
              if pred[0][1] > 0.5:
                  cur = mysql.connection.cursor()
                  cur.execute("UPDATE info SET prediction = 'cancer' WHERE id = ( SELECT MAX(id) FROM info )")
                  mysql.connection.commit()
                  cur.close()
                  return render_template('img.html',cancer=cancer)
              else:
                  cur = mysql.connection.cursor()
                  cur.execute("UPDATE info SET prediction = 'normal'  WHERE id = ( SELECT MAX(id) FROM info )")
                  mysql.connection.commit()
                  cur.close()
                  return  render_template('img.html',normal=normal)
    return render_template('img.html')

if (__name__=="__main__"):
    app.run(debug=True)