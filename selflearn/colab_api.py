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
# driver.get('https://colab.research.google.com/drive/1GPgQZGxFgQp9vJbaLJTp9rB-PEhECbXO?usp=sharing')
# driver.get('https://accounts.google.com/signin/v2/identifier?passive=true&continue=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1GPgQZGxFgQp9vJbaLJTp9rB-PEhECbXO&ec=GAZAqQM&flowName=GlifWebSignIn&flowEntry=ServiceLogin')
# https://accounts.google.com/signin/v2/identifier?passive=true&continue=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1GPgQZGxFgQp9vJbaLJTp9rB-PEhECbXO&ec=GAZAqQM&flowName=GlifWebSignIn&flowEntry=ServiceLogin


time.sleep(3)

print(driver.title)
# ctrl-F9
# run cell # command="runall"
# document.getElementById(":1v").click()
setup='''
document.getElementsByTagName("colab-run-button")[0].click()
'''
driver.execute_script(setup)
# document.getElementById("identifierNext").click()

time.sleep(3)

print(driver.title)
# pop up not authored by google
popup='''
document.getElementById("ok").click()
'''
# driver.execute_script(popup)


time.sleep(3)

while True:
    print(driver.title)
    high=input("here: ")
    looprun='''
    document.getElementsByTagName("colab-run-button")[1].click()
    img=document.getElementsByClassName("output_subarea output_image").item(0).children[0].src
    document.getElementsByClassName("view-lines").item(1).children[11].children[0].children[0].innerText={}
    '''.format("# "+high)
    driver.execute_script(looprun)
    # output_subarea output_image
    if high=="l":
        print(img)












# driver.quit()



# time.sleep(5) # Let the user actually see something!
# search_box = driver.find_element_by_name('q')
# search_box.send_keys('ChromeDriver')
# search_box.submit()
# time.sleep(5) # Let the user actually see something!
# driver.quit()
