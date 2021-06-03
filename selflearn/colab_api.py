from selenium import webdriver
from selenium_stealth import stealth
import time

# https://stackoverflow.com/questions/66209119/automation-google-login-with-python-and-selenium-shows-this-browser-or-app-may
options = webdriver.ChromeOptions()
# WINDOW_SIZE = "1280,720"
# options.add_argument("--window-size=%s" % WINDOW_SIZE)
# chrome_options.add_argument("--window-size=1920,1080")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument('--disable-blink-features=AutomationControlled')

# options.add_argument("--disable-web-security")
# # options.add_argument("--user-data-dir=true")
# options.add_argument("--allow-running-insecure-content")
options.add_argument('--user-data-dir=F:/selflearn/user-data')

# chrome_options.add_argument("user-data-dir=C:/Users/matth/AppData/Local/Google/Chrome/User Data/") #for same as local own
CHROMEDRIVER_PATH = 'C:/Users/matth/AppData/Local/Programs/Python/Python37/Lib/site-packages/selenium/webdriver/chromedriver.exe'

driver = webdriver.Chrome(CHROMEDRIVER_PATH, options=options)
# driver = webdriver.Chrome(CHROMEDRIVER_PATH, options=options, chrome_options=options)

stealth(driver,
    languages=["en-US", "en"],
    vendor="Google Inc.",
    platform="Win32",
    webgl_vendor="Intel Inc.",
    renderer="Intel Iris OpenGL Engine",
    fix_hairline=True,
    )

driver.get('https://colab.research.google.com/drive/1GPgQZGxFgQp9vJbaLJTp9rB-PEhECbXO')

# time.sleep(3)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
myElem = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, ':1x')))


print(driver.title)
# ctrl-F9
# run cell # command="runall"
# document.getElementById(":1v").click()
setup='''
document.getElementsByTagName("colab-run-button")[0].click()
'''
# driver.execute_script(setup)
# document.getElementById("identifierNext").click()

# time.sleep(3)


# if EC.presence_of_element_located((By.ID, ':1x'): # pop up not authored by google
# driver.execute_script("document.getElementById('ok').click()")
# ok=driver.find_element_by_xpath("//div[@id='ok']")

# <colab-dialog class="yes-no-dialog"
# <paper-dialog role="dialog"
# #shadow-root (open)
# usage limits usage-limit


# time.sleep(3)
import sys
while True:
    print(driver.title)
    high=input("here: ")
    # img=document.getElementsByClassName("output_subarea output_image").item(0).children[0].src
    looprun='''
    document.getElementsByTagName("colab-run-button")[1].click()
    document.getElementsByClassName("view-lines").item(1).children[11].children[0].children[0].innerText="{}"
    '''.format("# "+high)
    # driver.execute_script(looprun)

    driver.find_element_by_xpath("//colab-run-button/").click()
    driver.find_element_by_xpath("(//div[@class='view-lines'])[2]/div[11]/span/span").get_attribute("innerText")='# '+high



    # WebDriverWait(driver, 7).until(not EC.presence_of_element_located((By.CLASS_NAME, 'cell-execution focused animating running')))
    # cell-execution focused animating running



    # document.getElementsByClassName('output_subarea output_image').item(0).children[0].src #only if clicked within iframe
    iframe=driver.find_element_by_xpath("//div[@class='outputview']/iframe")
    driver.switch_to.frame(iframe)

    wait = WebDriverWait(driver, 1)
    # WebDriverWait(driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@id='textarea-WYSIWYG_ifr']")))
    # WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//body[@id='tinymce' and @class='mce-content-body']/p"))).send_keys("anynaynanya")


    img = driver.find_element_by_xpath("//div[@class='output_subarea output_image']/img")
    img = img.get_attribute("src")
    print(img)

    # driver.switch_to.default_content()
    # print("nope",sys.exc_info())


    # img=driver.execute_script("return document.getElementsByClassName('output_subarea output_image').item(0).children[0].src[get_img.index(':')+1:]")
    # get_img = document.getElementsByClassName('output_subarea output_image').item(0).children[0].src
    # get_img = get_img[get_img.index(':')+1:] # this will strip off 'javascript:' from the string name, so now 'func_name' contains 'onEditNumber();'
    # img = driver.execute_script('return ' + get_img)

    # print(img)



# search_box = driver.find_element_by_name('q')
# search_box.send_keys('ChromeDriver')
# search_box.submit()
# driver.quit()
