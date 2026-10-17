"""Fixed, synthetic evaluation dataset.

Ground truth for every case is computed from these rows with plain Python
(csv + statistics), never with Pandas.
"""

import csv
import io

SALES_CSV = """order_id,region,product,units,gross_revenue,net_revenue,channel
1,North,Pen,10,100,90,Online
2,North,Book,5,250,230,Store
3,South,Pen,20,200,180,Online
4,South,Book,,300,270,Store
5,East,Pen,7,70,65,Online
6,East,Book,3,150,140,Store
7,West,Pen,12,120,110,Online
8,West,Book,8,400,360,Online
9,North,Pen,15,150,135,Store
10,South,Pen,9,90,80,Online
11,East,Book,6,300,280,Online
12,West,Pen,4,40,35,Store
"""


def load_rows() -> list[dict]:
    return list(csv.DictReader(io.StringIO(SALES_CSV)))


def csv_bytes() -> bytes:
    return SALES_CSV.encode("utf-8")