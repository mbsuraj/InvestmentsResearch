from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser
import logging
import os
logging.basicConfig(level=logging.DEBUG)

class ExpenseLoader:

    def __init__(self):
        self.sObj = Splitwise(os.environ.get("CONSUMER_KEY"), os.environ.get("CONSUMER_SECRET"), api_key=os.environ.get("SPLITWISE_API_KEY"))
        # owner_id = self.sObj.getCurrentUser().id
        # self.owner = ExpenseUser()
        # self.owner.setId(owner_id)
        # # tenant_id = [friend for friend in self.sObj.getFriends() if friend.id==29057780][0]
        # self.tenant = ExpenseUser()
        # self.tenant.setId(os.environ.get("TENANT_ID"))
        # self.expense = Expense()

    def add_expense(self, amount: str, title="Water Bill"):
        expense = Expense()
        expense.setGroupId("87028983") # TODO: check actual group ID
        expense.setSplitEqually()
        expense.setCost(amount)
        created_expense, errors = s.createExpense(expense)
        # self.expense.setCost(amount)
        # self.expense.setDescription(title)
        # users = []
        # self.owner.setPaidShare(amount)
        # self.owner.setOwedShare("0.00")
        # # self.expense.setReceipt("/Users/naman/receipt.jpg")
        # self.tenant.setPaidShare("0.00")
        # self.tenant.setOwedShare(amount)
        # users.append(self.owner)
        # users.append(self.tenant)
        # self.expense.setUsers(users)
        # expense, errors = self.sObj.createExpense(self.expense)
        print(expense.getId())
# expense, errors = sObj.createExpense(expense)
# print(expense.getId())
# sObj.deleteExpense("3358074034")
