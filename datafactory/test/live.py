from ib_insync import IB, Contract, util
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=28998987, readonly=True, timeout=3.0)

# Replace conId and exchange with the one you resolved above (example uses the Contract object directly)
contract = Contract(conId=69067924, exchange='SMART')  # <- replace 12345678

# Request streaming market data (empty generic ticks list); snapshot=False to stream
ticker = ib.reqMktData(contract, '', snapshot=False, regulatorySnapshot=False)

# Print updates as they arrive
def onTicker(t):
    # t contains .bid, .ask, .last, .bidSize, .askSize, .time etc.
    print(t)

ib.pendingTickersEvent += onTicker

# keep running (or integrate into your event loop)
ib.run()
