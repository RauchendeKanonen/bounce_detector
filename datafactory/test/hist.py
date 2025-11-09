# historical ticks (useful when IB supports historical tick calls)
from ib_insync import IB, Contract
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=23132213, readonly=True, timeout=3.0)

contract = Contract(conId=69067924, exchange='SMART')  # replace

# Example: request last 1000 historical trade ticks
ticks = ib.reqHistoricalTicks(contract,
                              startDateTime='20251106 09:00:00',
                              endDateTime='20251106 16:00:00',
                              whatToShow='BID_ASK', useRth=False, numberOfTicks=1000)

for t in ticks:
    print('QUOTE', t.time, t.priceBid, t.sizeBid, t.priceAsk, t.sizeAsk)
