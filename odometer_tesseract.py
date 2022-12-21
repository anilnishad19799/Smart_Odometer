import cv2 
import pytesseract
from PIL import Image

Get_Image = Image.open('2.jpeg')

pytesseract.pytesseract.tesseract_cmd=r"C:/Users/aniln/AppData/Local/Tesseract-OCR/tesseract.exe"
custom_config = r'--oem 3 --psm 6 outputbase digits'
Get_Captcha = pytesseract.image_to_string(Get_Image)

print(Get_Captcha)
