from expenseLoader import ExpenseLoader
from gmailScraper import GmailScraper
import os


el = ExpenseLoader()
gs = GmailScraper()

amount = gs.scrape_bill()
el.add_expense(amount, os.environ.get("JOB"))
