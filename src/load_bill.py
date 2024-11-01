from expenseLoader import ExpenseLoader
from gmailScraper import GmailScraper
import os


el = ExpenseLoader()
gs = GmailScraper()

if os.environ.get("JOB") == "Rent":
    amount = 2750
elif os.environ.get("JOB") == "Internet Bill":
    amount = 51.22
elif os.environ.get("JOB") == "Electricity Bill":
    from srpScraper import SrpScraper
    ss = SrpScraper("https://www.srpnet.com/")
    amount = ss.get_bill()
elif os.environ.get("JOB") == "Water Bill":
    from waterBillScraper import WaterBillScraper
    ws = WaterBillScraper("https://ipn2.paymentus.com/cp/tmpp?lang=en")
    amount = ws.get_bill()
else:
    amount = gs.scrape_bill()
el.add_expense(amount, os.environ.get("JOB"))