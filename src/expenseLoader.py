from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser
import logging
import os
logging.basicConfig(level=logging.DEBUG)

class ExpenseLoader:

    def __init__(self):
        self.sObj = Splitwise(os.environ.get("CONSUMER_KEY"), os.environ.get("CONSUMER_SECRET"), api_key=os.environ.get("SPLITWISE_API_KEY"))
        owner_id = self.sObj.getCurrentUser().id
        self.owner = ExpenseUser()
        self.owner.setId(owner_id)
        # # tenant_id = [friend for friend in self.sObj.getFriends() if friend.id==29057780][0]
        # self.tenant = ExpenseUser()
        # self.tenant.setId(os.environ.get("TENANT_ID"))
        # self.expense = Expense()

    def add_expense(self, amount: str, title="Shared Bill", group_id=os.environ.get("TENANT_ID")):
        expense = Expense()
        expense.setCost(amount)
        expense.setDescription(title)
        expense.setGroupId(group_id)   # ✅ attach to group

        # Payer (you)
        self.owner.setPaidShare(amount)
        self.owner.setOwedShare("0.00")

        # Get group members (excluding you)
        if group_id:
            members = self.sObj.getGroup(group_id).members
        else:
            members = self.sObj.getFriends()

        # Filter out yourself
        members = [m for m in members if str(m.id) != str(self.owner.getId())]

        share = round(float(amount) / len(members), 2)
        adj = round(amount - share*len(members), 2)
        share_list = [share]*len(members)
        share_list[-1] += adj
        users = [self.owner]

        # Add all other members as owing equally
        for m, share in zip(members, share_list):
            u = ExpenseUser()
            u.setId(m.id)
            u.setPaidShare("0.00")
            u.setOwedShare(str(share))
            users.append(u)

        expense.setUsers(users)

        expense, errors = self.sObj.createExpense(expense)

        if errors:
            logging.error(f"Error creating expense: {errors.getErrors()}")
        else:
            logging.info(f"Expense created with ID {expense.getId()}")
            print(expense.getId())

# expense, errors = sObj.createExpense(expense)
# print(expense.getId())
# sObj.deleteExpense("3358074034")
