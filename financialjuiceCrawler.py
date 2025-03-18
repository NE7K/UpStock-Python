# note 필수 import part
from selenium import webdriver

# note 
# Keys = enter, ecs를 입력하고 싶을 때
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

# todo 어느 부분에서 사용되는지 체크
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# note load env file and keyword 'os'
from dotenv import load_dotenv
import os

# note 편의성 부분 - copy file or dir
import pyperclip

# .env file load
load_dotenv()

# selenium high version에서는 경로를 지정해줄 필요가 없음
openWindow = webdriver.ChromeOptions()























# info 종료 방지
input('enter')