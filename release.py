from kivymd.app import MDApp
from kivymd.uix.dialog import *
from kivymd.uix.button import *
from kivymd.uix.snackbar import Snackbar, BaseSnackbar
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import OneLineListItem
from kivymd.uix.swiper.swiper import MDSwiperItem
from kivymd.uix.behaviors.magic_behavior import MagicBehavior
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock
from ultralytics import YOLO
import json
import webbrowser
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import time
import os
import psutil
import random
import ctypes
from PyCameraList.camera_device import list_video_devices

# 常數
camera_list = [] # 相機list
dir = "" # 路徑
dialog = None # info初始
CL = [] # pre_class
is_video = False # 選擇Img/Cmaera
al_cls = [] # pre_all_classes
t_dict = {} # pre_dict
en_clss = {} # 空英文label字典
zh_clss = {} # 空中文label字典
url = "" # url_cls
history_class = []
url_index = 0
op_ch = 0 # choose Image / Video / Camera

animal = None # 動物swiper
manual = None # 使用手冊swiper
now = None # 現在的swiper

# 檔案路徑
model = YOLO("best.pt")
en_labels = "en_labels.txt"
zh_labels = "zh_labels.txt"
config = "config.json"

# 讀取設定檔
with open(config, encoding="utf-8") as f: # configuration
    setting_config = json.load(f)
language = setting_config["status"][0] # 中英
confidence = setting_config["status"][1] # 相似度
window_size = setting_config["status"][2] # 視窗大小
camera = setting_config["status"][3] # 選擇相機
cap = cv2.VideoCapture(camera) # 0:攝像頭1 / 1:攝像頭2
cap.read() # 預處理camera
with open(en_labels, encoding="utf-8") as f: # 英文label
    for index, line in enumerate(f):
        en_clss[index] = line.rstrip()
with open(zh_labels, encoding="utf-8") as f: # 中文label
    for index, line in enumerate(f):
        zh_clss[index] = line.rstrip()

def Pre(self, image, predictions): # 分析辨識結果與畫預測框
    global en_clss, zh_clss, language, url, history_class, url_index, al_cls, t_dict, is_video
    x, y = image.shape[1], image.shape[0] # img size
    for prediction in predictions: # 分析輸出結果
        REG = []
        # i = 0
        clss = prediction.names
        nowboxes = prediction.boxes.xywh
        nowclss = list(map(lambda j : float(j), prediction.boxes.cls))
        nowconf = list(map(lambda k : float(k),prediction.boxes.conf))
        if len(nowclss) > 0:
            al_cls.append(nowclss)
        print(is_video)
        if is_video == False and len(nowconf) > 0: # 處理圖像類別
            for index, cls in enumerate(nowclss):
                if language == "EN": url = f"{en_clss[int(cls)]}" #class
                else: url = f"{zh_clss[int(cls)]}" #class
                if url not in history_class:
                    history_class[index] = url
                print(history_class)
            
        elif len(al_cls) == 60 and is_video == True: # 處理影片類別並計算平均出現率
            avg_cls = round(len([i for j in al_cls for i in j]) / len(al_cls)) # 平均抓取class的數量
            for index, items in enumerate(al_cls):
                compare = len(items) - avg_cls
                if compare < 0: items += ([a for a in range(-1, compare-1, -1)]) #比較小補負數
                elif compare > 0: items = items[:-compare]#比較大的取前面
                print(compare)
                print(items)
                for item in items:
                    if item >= 0:
                        t_dict[item] = t_dict.get(item,0) + 1 #統計
                print(t_dict)
            for _ in range(len(t_dict)):
                key = max(t_dict, key=t_dict.get)
                t_max_value = t_dict.pop(key)
                count = int(t_max_value / len(al_cls) + .5)
                print(key, count)
                if count == 0: continue
                avg_cls -= count
                if avg_cls <= 0: break
            al_cls = []
            t_dict = {}
            if language == "EN": url = f"{en_clss[int(key)]}" #class
            else: url = f"{zh_clss[int(key)]}" #class
        if url not in history_class or is_video == False: # 判斷是否在清單裡面
            print(url)
            if is_video:
                if url_index > 9:
                    url_index = 0
                print(history_class) # test
                history_class[url_index] = url
                url_index += 1
            for index, class_name in enumerate(history_class):
                for child in self.ids.history_list.children:
                    if child.id == f"history_list_item{index}": child.text = f"[size=18][font=msjh.ttc]{history_class[index]}[/font][/size]"

        for i, box in enumerate(nowboxes): # 畫boundbox
            boundcolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # 線的顏色
            REG.append(list(map(lambda x : float(x), box)))
            x1, y1, x2, y2 = int(REG[i][0] - REG[i][2]/2), int(REG[i][1] - REG[i][3]/2), int(REG[i][0] + REG[i][2]/2), int(REG[i][1] + REG[i][3]/2) # yolo是以框的中心點x, y，要轉換成opencv的x, y, x2, y2
            bbox_rate = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 100
            img_rate = math.sqrt(x ** 2 + y ** 2) / 100
            bound_len = int(3 * (bbox_rate + img_rate)) # 線的長度
            bound_thickness = int(0.8 * img_rate) # 線的粗細
            print(bbox_rate, img_rate)
            dict = {(x1, y1):[bound_len, bound_len], (x1, y2):[bound_len, -bound_len], (x2, y1):[-bound_len, bound_len], (x2, y2):[-bound_len, -bound_len]}
            for k, v in dict.items(): # 四角框(左上>右上>左下>右下)
                # print(k, v)
                cv2.line(image, (k[0], k[1]), (k[0]+v[0], k[1]), boundcolor, bound_thickness) # 直線
                cv2.line(image, (k[0], k[1]), (k[0], v[1]+k[1]), boundcolor, bound_thickness) # 橫線
            # Label
            fontcolor = (0, 0, 0) # 文字顏色
            if language == "EN": # en or zh-tw
                a, b = cv2.getTextSize(f"{en_clss[int(nowclss[i])]} {nowconf[i] * 100 :.1f}%", cv2.FONT_HERSHEY_SIMPLEX, 0.025*bound_len, int(0.08*bound_len)) # 偵測文字長度
                cv2.rectangle(image, (x1+bound_thickness, y1+bound_thickness), (x1 + a[0], y1 + a[1] + bound_thickness * 2), (255, 255, 255), -1) # 畫上文字背景
                cv2.putText(image, f"{en_clss[int(nowclss[i])]} {nowconf[i] * 100 :.1f}%", (x1 + bound_thickness, y1 + a[1] + bound_thickness), cv2.FONT_HERSHEY_SIMPLEX, 0.025*bound_len, fontcolor, int(0.08*bound_len)) # 寫上文字(名稱、相似度)                
            else:
                print(f"{zh_clss[int(nowclss[i])]}")
                a, b = cv2.getTextSize(f"{zh_clss[int(nowclss[i])]} {nowconf[i] * 100 :.1f}%", cv2.FONT_HERSHEY_SIMPLEX, 0.04*bound_len, int(0.1*bound_len)) # 偵測文字長度
                cv2.rectangle(image, (x1+bound_thickness, y1+bound_thickness), (x1 + int(a[0]/2), y1 + a[1] + bound_thickness), (255, 255, 255), -1) # 畫上文字背景
                img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                font = ImageFont.truetype("./font/msjh.ttc", int(0.8*bound_len))
                position = (x1 + bound_thickness, y1) # text position
                str = f"{zh_clss[int(nowclss[i])]} {nowconf[i] * 100 :.1f}%"
                draw = ImageDraw.Draw(img_PIL)
                draw.text(position, str, font=font, fill=fontcolor)
                image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
            # i += 1
    return image

class Setting_Content(MDBoxLayout): # 設定介面內容
    Language_text = StringProperty('')
    EN_selected = StringProperty('')
    CN_selected = StringProperty('')
    Confidence_text = StringProperty('')
    Confidence_selected = StringProperty('')
    Confidence_value = StringProperty('')
    Windowsize_text = StringProperty('')
    Windowsize_selected = StringProperty('')
    Camera_text = StringProperty('')
    Camera_selected = StringProperty('')

    def ch_language(self, selected_instance_chip): # 更改語言觸發
        global language
        print(self.EN_selected)
        selected_instance_chip.icon_left = "check-circle-outline"
        if selected_instance_chip.text == "English": self.ch_lg = "EN"
        else: self.ch_lg = "CN"
        for instance_chip in self.ids.EN_CH.children:
            if instance_chip != selected_instance_chip:
                instance_chip.icon_left = ""
        language = self.ch_lg

    def ch_confidence(self, instance): # 更改相識度
        global confidence
        self.ids.conf_value.text = f"{int(instance.value)}%"
        confidence = np.around(instance.value / 100, 3)

    def open_window_size(self, instance): # 開啟視窗drop down
        global setting_config, window_size
        self.menu = None
        menu_items = [
            {
                "viewclass" : "OneLineListItem",
                "text" : f"[font=./font/msjh.ttc]{i}[/font]",
                "height" : dp(56),
                "on_release": lambda x=f"{i}": self.ch_window_size(x),
            } for i in setting_config["Window_size"]
        ]
        if not self.menu:
            self.menu = MDDropdownMenu(
                items = menu_items,
                position = "bottom",
                max_height=dp(224),
                width_mult= 2,
            )
        self.menu.caller = instance
        self.menu.open()

    def pre_size(self, instance): # 處理視窗大小
        self.factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
        self.screenx = int(ctypes.windll.user32.GetSystemMetrics(0))
        self.screeny = int(ctypes.windll.user32.GetSystemMetrics(1))
        x, y = map(lambda x : int(x), instance.split("x"))
        Window.minimum_width, Window.minimum_height = int(800 / self.factor), int(600 / self.factor)
        Window.size = (int(x / self.factor), int(y / self.factor))
        Window.left = int((self.screenx - Window.size[0])/2)
        Window.top = int((self.screeny - Window.size[1])/2)

    def ch_window_size(self, instance): # 選擇視窗大小
        global window_size
        self.pre_size(instance)
        self.ids.size_value.text = instance
        window_size = instance
        self.menu.dismiss()

    def open_camera_select(self, instance): # 開啟相機drop down
        global setting_config, window_size, camera_list
        self.menu = None
        menu_items = [
            {
                "viewclass" : "OneLineListItem",
                "text" : f"[font=./font/msjh.ttc]{name}[/font]",
                "height" : dp(56),
                "on_release": lambda x=[i, f"{name}"]: self.ch_camera(x),
            } for i, name in camera_list
        ]
        if not self.menu:
            self.menu = MDDropdownMenu(
                items = menu_items,
                position = "bottom",
                max_height=dp(224),
                width_mult= 5,
            )
        self.menu.caller = instance
        self.menu.open()

    def ch_camera(self, instance): # 選擇相機
        global camera, cap
        print(instance)
        camera = instance[0]
        self.ids.camera_value.text = instance[1]
        cap = cv2.VideoCapture(camera) # 0:攝像頭1 / 1:攝像頭2
        self.menu.dismiss()

    def create_swiper(self): # 創建動物 / 使用手冊swiper
        global en_clss, zh_clss, language, animal, manual, now
        animal = Factory.Animal()
        manual = Factory.Manual()
        now = animal
        for index, source in enumerate(en_clss.values()):
            swiperitem = Factory.AnimalSwiper()
            swiperitem.ids.img.source = f"./animal/{source}.jpg"
            if language == "EN": swiperitem.ids.label.text = f"[font=./font/msjh.ttc]{source}[/font]"
            else: swiperitem.ids.label.text = f"[font=./font/msjh.ttc]{list(zh_clss.values())[index]}[/font]"
            animal.add_widget(swiperitem)
        for source in os.listdir(f"./manual/{language}/"):
            swiperitem = Factory.ManualSwiper()
            swiperitem.ids.img.source = f"./manual/{language}/{source}"
            manual.add_widget(swiperitem)

class CustomSnackbar(BaseSnackbar): # 提示Info
    text = StringProperty(None) # 文字
    icon = StringProperty(None) # 圖示
    font_size = NumericProperty("30sp")
    duration = 0.05 # 持續時間(seconds)

class AnimalSwiper(MagicBehavior, MDSwiperItem): # 動物Swiper
    global language
    def open_url(self):
        str1 = self.ids.label.text.split("[")
        str2 = str1[1].split("]")
        search_url = f"{str2[1]}"
        if language == "EN": webbrowser.open(f"https://en.wikipedia.org/wiki/{search_url}", new=1, autoraise=True)
        else: webbrowser.open(f"https://zh.wikipedia.org/zh-tw/{search_url}", new=1, autoraise=True)

class Mylayout(MDBoxLayout): # Main Screen
    Home_btn = StringProperty('')
    Choose_btn = StringProperty('')
    Camera_btn = StringProperty('')
    Setting_btn = StringProperty('')
    language_title = StringProperty('')
    confidence_title = StringProperty('')
    windowsize_title = StringProperty('')
    camera_title = StringProperty('')
    ST_accept = StringProperty('')
    Detect_btn = StringProperty('')
    List_title = StringProperty('')
    Wiki_btn = StringProperty('')
    Animal_title = StringProperty('')
    Manual_btn = StringProperty('')
    Back_btn = StringProperty('')
    sour = ()
    rootpath = True

    @classmethod
    def widgets_init_language(cls, self): # 同步修改語言
        global setting_config, language, window_size
        widgets = [key for key, value in cls.__dict__.items() if not key.startswith('__') and not callable(value)][:-3]
        for index, name in enumerate(widgets):
            items = [
                list(setting_config["UI_language"].values())[index],
                list(setting_config["UI_language"].keys())[index]
            ]
            if setting_config["status"][0] == "EN": text = items[0]
            else: text = items[1]
            if index in [3, 13, 14]: attribute = f"[font=msjh.ttc]{text}[/font]" # 需要在text設中文字體的文字
            else: attribute = text
            setattr(self, name, attribute)

    def __init__(self, **kwargs): # 初始UI介面語言
        global setting_config, language, window_size
        super(Mylayout, self).__init__(**kwargs)
        self.widgets_init_language(self)

    def switch_tabs(self, instance_tab): # 選擇導覽列觸發
        # 對應的Screen初始化
        global setting_config, url, al_cls, t_dict, cap, op_ch, url_index, is_video, op_ch, history_class, url_index
        try: self.event.cancel()
        except: pass
        self.search_url = ""
        url = ""
        al_cls = []
        t_dict = {}
        self.ids.FPS_label.text = f"FPS："
        if instance_tab.text == self.Home_btn: # home init
            self.ids.screen_manager.current = instance_tab.name
        elif instance_tab.text == self.Choose_btn: # choose img/video init
            self.ids.screen_manager.current = instance_tab.name
            self.partitions = psutil.disk_partitions(all=True) # 列出所有disk detail
            self.manager_open = False
            if self.rootpath:
                self.ids.filechooser.rootpath = self.partitions[0].device
                self.ids.drop_item.text = self.partitions[0].device
                self.rootpath = False
        elif instance_tab.text == self.Camera_btn: # detect init
            r, img = cap.read()
            if not r:
                if language == "EN": self.Info(self, "Not connected to camera yet.", "Please connect to camera.")
                else: self.Info(self, "尚未連接上Camera", "請連接上Camera")
            else:
                op_ch = 3
                is_video = True
                self.detect_init()
                    
    def detect_init(self): # 辨識前的初始化
        global setting_config, dir, history_class, url_index, is_video
        self.ids.List_title.text = f"{self.List_title}"
        self.ids.history_list.clear_widgets()
        history_class = ["" for _ in range(10)] # reset the history
        url_index = 0
        self.fps = 0
        self.frame_count = 0

        if op_ch == 1: # 圖片辨識
            self.Image_detect()
            self.ids.FPS_label.text = f"FPS：1"
        elif op_ch == 2 or op_ch == 3: # 影片辨識 / 相機辨識
            if op_ch == 2: self.video = cv2.VideoCapture(dir)
            self.start_time = time.time()
            self.event = Clock.schedule_interval(self.Video_detect, 1 / 30)
    
        for index ,i in enumerate(history_class): # 建立history list items
            self.ids.history_list.add_widget(
                OneLineListItem(
                    id = f"history_list_item{index}", 
                    text = f"[size=18][font=msjh.ttc]{i}[/font][/size]",
                    on_touch_down = self.reset_background,
                    on_press = self.choose_history
                )
            )
        self.ids.screen_manager.current = "screen3"

#================ Home ================

    def Info(self, instance, T1, T2): # 提示視窗
        # 按鈕會回傳兩個值，T1和T2是我們輸出的字
        self.dialog = MDDialog(
            title = f"[font=msjh.ttc]{T1}[/font]",
            text = f"[font=msjh.ttc]{T2}[/font]",
            buttons = [
                MDRaisedButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    on_release = self.close
                )
            ]
        )
        self.dialog.open()
    
    def close(self, instance):
        self.dialog.dismiss()

    def ch_swipers(self, instance): # 選擇menual / animals
        global animal, manual, now
        if self.ids.Manual_btn.icon == "arrow-u-down-left-bold":
            self.ids.Manual_btn.icon = "book-open-blank-variant"
            self.ids.animal_title.opacity = 1
            self.ids.animal_title.size_hint = (1, .1)
            self.ids.Manual_btn.tooltip_text = self.Manual_btn
            self.ids.change_swiper.remove_widget(manual)
            self.ids.change_swiper.add_widget(animal)
            now = animal
        elif not self.ids.Manual_btn.icon == "arrow-u-down-left-bold":
            self.ids.Manual_btn.icon = "arrow-u-down-left-bold"
            self.ids.animal_title.opacity = 0
            self.ids.animal_title.size_hint = (1, .001)
            self.ids.Manual_btn.tooltip_text = self.Back_btn
            self.ids.change_swiper.remove_widget(animal)
            self.ids.change_swiper.add_widget(manual)
            now = manual

    def swipe_left_right(self, direction): # 控制siwper上下頁
        global now
        if direction == "left":
            now.swipe_left()
        elif direction == "right":
            now.swipe_right()

    def open_setting(self, instance): # 開啟dialog settings
        global language, confidence, window_size, camera, camera_list
        camera_list = list_video_devices()
        print(camera_list)
        if camera_list != []:
            available_camera = f"{camera_list[camera][1]}"
        elif camera_list == [] and language == "CN": available_camera = "無可用的相機"
        elif camera_list == [] and language == "EN": available_camera = "No camera"
        Setting_Content.Language_text = self.language_title
        if language == "EN":
            Setting_Content.EN_selected = "check-circle-outline"
            Setting_Content.CN_selected = ""
        else:
            Setting_Content.EN_selected = ""
            Setting_Content.CN_selected = "check-circle-outline"
        Setting_Content.Confidence_text = self.confidence_title
        Setting_Content.Confidence_selected = int(confidence*100)
        Setting_Content.Confidence_value = f"{int(confidence*100)}%"
        Setting_Content.Windowsize_text = self.windowsize_title
        Setting_Content.Windowsize_selected = window_size
        Setting_Content.Camera_text = self.camera_title
        Setting_Content.Camera_selected = available_camera

        self.setting_dialog = None
        if not self.setting_dialog:
            self.setting_dialog = MDDialog(
                title = f"{self.Setting_btn}",
                type = "custom",
                content_cls = Setting_Content(),
                buttons = [
                    MDRaisedButton(
                    text = f"[font=./font/msjh.ttc]{self.ST_accept}[/font]",
                    md_bg_color = "lightblue",
                    text_color = "black",
                    on_release = self.accept
                    )
                ]
            )
        self.setting_dialog.open()

    def accept(self, instance): # 確認設定並套用更改
        global setting_config, language, confidence, window_size, camera, config, animal, manual, now
        print(setting_config)
        setting_config["status"][0] = language
        setting_config["status"][1] = confidence
        setting_config["status"][2] = window_size
        setting_config["status"][3] = camera
        print(setting_config)
        with open(config, "w",  encoding="utf8") as f: # 儲存設定檔
            json.dump(setting_config, f, indent=4, ensure_ascii = False)
        if language == "EN": Object_DetectionApp().underInfo("Language：English")
        else: Object_DetectionApp().underInfo("語言：Chinese")
        self.ids.change_swiper.clear_widgets()
        Setting_Content().create_swiper()
        self.ids.Manual_btn.icon = "book-open-blank-variant"
        self.ids.Manual_btn.tooltip_text = self.Manual_btn
        self.ids.animal_title.opacity = 1
        self.ids.animal_title.size_hint = (1, .1)
        self.ids.change_swiper.remove_widget(manual)
        now = animal
        self.ids.change_swiper.add_widget(animal)
        self.setting_dialog.dismiss()
        self.widgets_init_language(self)

#================ Choose Image/Video ================

    def open_disks(self, instance): # 選擇硬碟
        partition = [i.device for i in self.partitions]
        self.menu = None
        menu_items = [
            {
                "viewclass" : "OneLineListItem",
                "text" : f"[font=./font/msjh.ttc]{i}[/font]",
                "height" : dp(56),
                "on_release": lambda x=f"{i}": self.open_c1(x),
            } for i in partition
        ]
        if not self.menu:
            self.menu = MDDropdownMenu(
                items= menu_items,
                width_mult= 1.5,
            )
        self.menu.caller = instance
        self.menu.open()
    
    def open_c1(self, instance): # 選擇硬碟觸發
        self.ids.filechooser.rootpath = instance
        self.ids.drop_item.text = instance
        self.menu.dismiss()

    def show_path(self, instance): # 更新目前路徑
        self.ids.search_path.text = self.ids.filechooser.path # 顯示目前路徑

    def search(self, instance): # 搜尋路徑
        self.ids.filechooser.path = self.ids.search_path.text # 變更路徑

    def selected(self,filename): # 選擇圖片/影片
        global dir
        try:
            dir = filename[0]
            print(dir)
            self.sour = os.path.splitext(dir) # 抓取副檔名
            print(self.sour)
            self.ids.MDNavigationDrawerHeader.title = f"[font=./font/msjh.ttc]{os.path.basename(dir)}[/font]"
            if self.sour[1] == ".jpg" or self.sour[1] == ".png" or self.sour[1] == ".jpeg" or self.sour[1] == ".jfif":
                img = cv2.imread(dir)
            if self.sour[1] == ".mav" or self.sour[1] == ".mp4":
                view = cv2.VideoCapture(dir)
                r, img = view.read()
            x, y = img.shape[1], img.shape[0] # img size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            buf = cv2.flip(img, 0).tobytes() # buffer
            self.texture = Texture.create(size = (x, y), colorfmt='rgb')
            self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.ids.my_image.texture = self.texture
            self.ids.nav_drawer.set_state("open") # 顯示右側預覽
            self.ids.filechooser.selection = [] # 清除選擇暫存
        except:
            self.ids.my_image.source = ""
    
    def ImgOD(self, instance): # 進行辨識
        global dir, model, url, url_index, op_ch, history_class, confidence, is_video, language
        url_index = 0
        self.sour = os.path.splitext(dir) # 抓取副檔名
        if self.sour[1] == ".mp4" or self.sour[1] == "mav":
            # 判斷是否為影片
            is_video = True
            print(dir) # 影片路徑
            # 預處理
            view = cv2.VideoCapture(dir)
            r, img = view.read()
            test = model.predict(img, conf=confidence)
            self.ids.out_detection.texture = self.texture
            op_ch = 2
            self.detect_init()

        elif self.sour[1] == ".jpg" or self.sour[1] == ".png" or self.sour[1] == ".jpeg" or self.sour[1] == ".jfif":
            # 判斷是否為圖片
            try:
                is_video = False
                self.ids.out_detection.color = (0, 0, 0, 1)
                op_ch = 1
                self.detect_init()
            except FileNotFoundError:
                if language == "EN": self.Info(self, "File not found", "Please check file paths.")
                else: self.Info(self, "檔案不存在或路徑不正確", "請檢查檔案路徑")

#================ Detect ================
            
    def RGBA_to_RGB(self, img): # 把RGBA圖片轉換為RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # (img)RGBA to (cv)BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (cv)BGR to (img)RGB
        return img

    def Image_detect(self): # 辨識圖片
        global dir, model, confidence
        try:
            img = cv2.imdecode(np.fromfile(dir,dtype=np.uint8), -1) # cv2讀取圖片
            print(img.shape)
            if img.shape[2] == 4: img = self.RGBA_to_RGB(img)
            test = model.predict(img, conf=confidence)
            image = Pre(self, img, test) # prediction
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (cv)BGR to (kivy)RGB
            buf = cv2.flip(img, 0).tobytes() # buffer
            x, y = img.shape[1], img.shape[0] # image size
            texture = Texture.create(size = (x, y), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.ids.out_detection.texture = texture
            self.ids.out_detection.color = (1, 1, 1, 1)
        except:
            pass

    def Video_detect(self, dt): # 辨識影片
        global op_ch, dir, cap, model, confidence, al_cls, t_dict
        try:
            if op_ch == 2: r, img = self.video.read()
            if op_ch == 3: r, img = cap.read()
            x, y = img.shape[1], img.shape[0] # img size
            test = model.predict(img, conf=confidence)
            image = Pre(self, img, test)
            self.frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            if elapsed_time >= 0:
                self.fps = self.frame_count / elapsed_time
                self.ids.FPS_label.text = f"FPS：{int(self.fps)}"
                print(self.fps)
            self.start_time = time.time()
            self.frame_count = 0
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            buf = cv2.flip(img, 0).tobytes() # buffer
            texture = Texture.create(size = (x, y), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.ids.out_detection.texture = texture
            self.ids.out_detection.color = (1, 1, 1, 1)
        except AttributeError:
            self.event.cancel()
            al_cls = []
            t_dict = {}

    def reset_background(self, instance, pos): # 重置history list items按鈕顏色
        instance.bg_color = (1, 1, 1, .01)

    def choose_history(self, instance): # 選擇history list items
        instance.bg_color = (1, 1, 1, .35)
        print(f"{instance.text}")
        if instance.text != "":
            str1 = instance.text.split("[")
            str2 = str1[2].split("]")
            self.search_url = f"{str2[1]}"
        else: self.search_url = ""

    def open_url(self, instance): # 開啟wiki
        global language
        if self.search_url != "":
            if language == "EN":
                webbrowser.open(f"https://en.wikipedia.org/wiki/{self.search_url}", new=1, autoraise=True)
            else: webbrowser.open(f"https://zh.wikipedia.org/zh-tw/{self.search_url}", new=0, autoraise=True)
        else: 
            if language == "EN": Object_DetectionApp().underInfo("Please select an animal name.")
            else: Object_DetectionApp().underInfo("請選擇名稱")

class Object_DetectionApp(MDApp):
    def build(self):
        global window_size, animal, manual, now
        Setting_Content().pre_size(window_size)
        self.theme_cls.theme_style_switch_animation = True
        self.theme_cls.theme_style = "Dark" # window reset color
        kv = Builder.load_file("release.kv")
        return kv

    def on_start(self):
        global animal
        Setting_Content.create_swiper(self)
        self.root.ids.change_swiper.add_widget(animal)

    def underInfo(self, T1):
        # 按鈕會回傳兩個值，T1和T2是我們輸出的字
        snackbar = CustomSnackbar(
            text=f"{T1}",
            icon="information",
            snackbar_x = "10dp",
            snackbar_y = f"{Window.height / 10}dp",
        )
        snackbar.size_hint_x = (
            Window.width - (snackbar.snackbar_x * 2)
        ) / Window.width
        self.theme_cls.theme_style = "Dark" # window reset color
        snackbar.open()
    
if __name__ == "__main__":
    Object_DetectionApp().run()