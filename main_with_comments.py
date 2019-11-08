
import numpy as np #иморрт нумпая для работы с np.array
from kivy.config import Config #испорт разных kivy объектов
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2 #иморт библиотеки для работы с изображениями

import time 
import cvlib as cv #импорт фреймворка с сверточной нейросетью для детекции объектов - Yolo3
from cvlib.object_detection import draw_bbox

#constans разные параметры, влияющие на работу программы

DELTA = 5      #Половина толщины рамки в пикселях
TRESH = 0    #Трешхолд, степень уверенности, начиная с которой объекты выделяются, д.ед
TARGET_OBJ = 'person' #целевой объект 
COLOR = [255,0,0]     #цвет рамки, RGB
DETECT_TIME = 5       #n, обновление рамки проиходит каждые n кадров 

#globals
iteration = DETECT_TIME
bbox, label, conf = [], [], []


# создание окна kivy
#объект Camera отвечает за захват изображения и его отображения с:
#объект Image является буфером в который выбрасываются обработанные изображения. Так транслируется обработанное видео

Builder.load_string('''
<Main>:
    padding : 30
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: True
    Image:
        id: buf
        source: '3.jpg'
        size: self.texture_size
''')


#заглушки
#должны лежать в папке с кодом
p1 = '1.jpg'
p2 = '2.jpg'


def change_pic(pic, lout):  #функция обновления картинки 
    if lout.ids['camera'].texture != None:  #если изображение есть
        #объект Camera имеет поле pixels в битовой кодировке. np.array(list()) переводит битовую строку в строку с 
        #значениями от 0 до 255(цвет). Пока это только "колбаска". reshape приводит ее к 3х мерному тензору размера 
        #разрешения камеры(480, 640) с 4 каналами(RGB + alpha). Альфа-канал исключаем. 
        output_image = np.array(list(lout.ids['camera'].texture.pixels), dtype=np.uint8).reshape(480, 640, 4)[:,:,:3]
        #нейросеть работает не быстро. Есть смысл искать объекты не каждый кадр. Информацию о 
        #найденных объектах кинем в глобальную область видимости
        global bbox, label, conf
        global iteration
        #Преобразование Гауса с ядром (15,15) для избавления от шума. Пока не работает.  Надо тестить.
        #img = cv2.GaussianBlur(img,(15,15),0)
        if iteration % DETECT_TIME == 0:  #пришло время обновить информацию о объектах
            #получим информацию от Yolo3. bbox - расположение объекта(левый верхний и правый нижний углы прямоугольника)
            #label - метка класса. conf - уверенность (0-1)
            bbox, label, conf = cv.detect_common_objects(output_image, confidence=TRESH, model='yolov3-tiny')
            #log_of_detction лог
            #print(label, conf) 
        iteration+=1
     
        for i in range(len(conf)):
            if label[i] == TARGET_OBJ and conf[i] >= TRESH:  #если уверенность превышает трешхолд
            #нарисуем прямоугольник вокруг найденного объекта
                output_image[bbox[i][1] - DELTA : bbox[i][1] + DELTA , bbox[i][0] : bbox[i][2]] = np.array(COLOR, dtype=np.uint8)
                output_image[bbox[i][3] - DELTA : bbox[i][3] + DELTA , bbox[i][0] : bbox[i][2]] = np.array(COLOR, dtype=np.uint8)
                output_image[bbox[i][1] : bbox[i][3] , bbox[i][0] - DELTA : bbox[i][0] + DELTA] = np.array(COLOR, dtype=np.uint8)
                output_image[bbox[i][1] : bbox[i][3] , bbox[i][2] - DELTA : bbox[i][2] + DELTA] = np.array(COLOR, dtype=np.uint8)
        #магия по переводу 3х мерного тензора обратно в формат, в котором хранятся изображения в kivy(Texture)
        image_texture = Texture.create(
            size=(output_image.shape[1], output_image.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(cv2.flip(output_image, 0).tostring(), colorfmt='rgb', bufferfmt='ubyte')
        #отобразим изображение. Закинем его в буфер Image с тегом "buf"(см.создание окна kivy).
        #стоит отметить, что изображение выводится без задержки с таймингом, соответсвующим захвату видео. Однако 
        #Как было сказано выше, ресурсы моего(!) компьютера не позволяют искать объекты каждый кадр. 
        #Обновление информации о расположении объектов происходит каждые  проиходит каждые DETECT_TIME(см. globals) кадров.
        pic.texture = image_texture
    Clock.schedule_once(lambda _: change_pic(pic, lout), 0)  #запланируем выполнить эту же функцию для следующего кадра.

    #создадим окно kivy с 2 заглушками
class Main(BoxLayout):
    def reloading(self):
        self.ids['buf'].source = '1.jpg'
    def reloading2(self):
        self.ids['buf'].source = '2.jpg'
    pass
#создадим приложение kivy
class WEEDEO(App):
    def build(self):
        blt = Main()
        change_pic(blt.ids['buf'], blt)
        return blt


#запустим приложение kivy
WEEDEO().run()
