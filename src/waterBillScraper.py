import os
import time
import json
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


class WaterBillScraper:
    def __init__(self, url="https://ipn2.paymentus.com/cp/tmpp?lang=en"):
        self.url = url
        self.driver = webdriver.Chrome(options=self._get_options())
        stealth(self.driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
                )
        with open(f'{os.getcwd()}/input/waterbill_xpath_dict.json') as json_file:
            self.xpaths = json.load(json_file)
        self.laptop_details = None

    def _get_options(self):
        # PROXY_STR = "111.222.111.222:1234"
        options = webdriver.ChromeOptions()
        options.add_argument("start-maximized")
        options.add_argument("--incognito")
        # options.add_argument('--proxy-server=%s' % PROXY_STR)
        options.add_argument("user-agent=THis")
        # options.add_argument("--headless")
        options.add_argument("disk-cache-size=0")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        return options

    def _refresh_driver(self):
        self.driver.delete_all_cookies()
        self.driver.quit()
        self.driver = webdriver.Chrome(options=self._get_options())
        stealth(self.driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
                )
        time.sleep(2)

    def get_bill(self):
        self.driver.get(self.url)
        time.sleep(5)
        username = self.driver.find_element(by=By.XPATH, value=self.xpaths["input"]["username"])
        pwd = self.driver.find_element(by=By.XPATH, value=self.xpaths["input"]["password"])
        username.send_keys(os.environ.get("SRP_USERNAME"))
        time.sleep(1)
        pwd.send_keys(os.environ.get("SRP_PWD"))
        time.sleep(1)
        login = self.driver.find_element(by=By.XPATH, value=self.xpaths["button"]["log_in"])
        login.click()
        time.sleep(20)
        try:
            bill = self.driver.find_element(by=By.XPATH, value=self.xpaths["output"]["bill"])
            return float(bill.text.replace("Amount Due", "").replace("$", "").strip())
        except NoSuchElementException:
            self.driver.get_screenshot_as_file(f"{os.getcwd()}/diagnostics/snapshot_for_debug.png")
            raise NoSuchElementException
