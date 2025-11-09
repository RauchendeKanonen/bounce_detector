from ib_insync import IB, Contract

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=11221, readonly=True, timeout=3.0)   # TWS default paper port; 4002/4001 for IB Gateway depending on setup

# Example: try XAUUSD (spot gold). If you prefer USGOLD try that symbol too.
c = Contract(symbol='XAUUSD', secType='CMDTY', exchange='SMART', currency='USD')

# Ask IB for details (this returns one or more ContractDetails with conId)
details = ib.reqContractDetails(c)
for d in details:
    print('ConId:', d.contract.conId, d.contract.localSymbol, d.contract.secType, d.contract.exchange)
# Pick the correct conId from the list you get back and use that Contract going forward.
