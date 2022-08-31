from selenium.webdriver import Firefox
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import os

# Create driver
service = Service(os.getcwd() + '\\geckodriver.exe')
options = FirefoxOptions()
driver = Firefox(service = service, options = options)

action = ActionChains(driver)
url = 'https://googledino.com/'
driver.get(url)


def press_space():
    action.key_down(Keys.SPACE)
    action.key_up(Keys.SPACE)
    action.perform()

def get_canvas():
    element = driver.find_element(By.CLASS_NAME, 'runner-canvas')
    return element
