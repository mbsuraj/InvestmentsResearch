from expenseLoader import ExpenseLoader
from gmailScraper import GmailScraper
import os


el = ExpenseLoader()
gs = GmailScraper()

if os.environ.get("JOB") == "Rent":
    amount = 2750
elif os.environ.get("JOB") == "Internet Bill":
    amount = 51.22
else:
    amount = gs.scrape_bill()
el.add_expense(amount, os.environ.get("JOB"))
