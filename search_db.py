#!/usr/bin/env python3
"""
find_pg_market_db.py

Discover local PostgreSQL servers, enumerate databases, and identify tables
that look like market data based on names and columns.

Usage:
  python find_pg_market_db.py --max-port 5450 --min-score 2
"""

import argparse
import getpass
import os
import re
import socket
import sys
from contextlib import contextmanager

# Try psycopg (v3) first, then psycopg2
DBLIB = None
try:
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore
    DBLIB = "psycopg3"
except Exception:
    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore
        DBLIB = "psycopg2"
    except Exception:
        print("Please install 'psycopg' (v3) or 'psycopg2' first, e.g.:")
        print("  pip install psycopg[binary]  # or: pip install psycopg2-binary")
        sys.exit(1)

DEFAULT_PORTS = list(range(5432, 5451))
SOCKET_DIRS = [None, "/var/run/postgresql", "/tmp", "localhost"]  # None = default unix socket

# Keywords that hint a table is market/finance-related
NAME_HINTS = [
    "market", "price", "prices", "trade", "trades", "quote", "quotes",
    "tick", "ticks", "ticker", "symbol", "symbols",
    "bar", "bars", "candle", "candles", "ohlc", "orderbook", "depth",
    "bid", "ask", "spread", "execution", "fill", "level2", "book",
    "fx", "forex", "crypto", "equity", "stock", "futures", "option", "deriv"
]

# Column names that are common in market data
COLUMN_HINTS = [
    "time", "timestamp", "ts", "datetime", "date",
    "ticker", "symbol", "instrument",
    "open", "high", "low", "close", "ohlc",
    "bid", "ask", "bid_size", "ask_size", "bidsize", "asksize",
    "volume", "vwap", "turnover", "notional",
    "price", "last", "mid", "mark",
    "exchange", "venue"
]

def port_open(host: str, port: int, timeout=0.1) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

@contextmanager
def connect_any(dsn_kwargs):
    """
    Context manager that yields a connection using either psycopg3 or psycopg2
    based on what's available.
    """
    conn = None
    try:
        if DBLIB == "psycopg3":
            # Convert None host properly; psycopg3 accepts host as None
            conn = psycopg.connect(**dsn_kwargs)
        else:
            # psycopg2: need to build a DSN string
            import psycopg2
            import psycopg2.extras
            params = []
            for k, v in dsn_kwargs.items():
                if v is None:
                    continue
                params.append(f"{k}={v}")
            dsn = " ".join(params)
            conn = psycopg2.connect(dsn)
        yield conn
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def dict_cursor(conn):
    if DBLIB == "psycopg3":
        return conn.cursor(row_factory=dict_row)
    else:
        return conn.cursor(cursor_factory=psycopg2.extras.DictCursor)  # type: ignore

def try_connect(host, port, user, dbname="postgres"):
    dsn = {
        "host": host,      # None means default unix socket for psycopg3
        "port": port,
        "user": user,
        "dbname": dbname,
        # Passwords will be sourced from .pgpass / env if needed
    }
    try:
        with connect_any(dsn) as conn:
            with dict_cursor(conn) as cur:
                cur.execute("SELECT version();")
                v = cur.fetchone()
                return True, v.get("version") if v else "unknown"
    except Exception as e:
        return False, str(e)

def list_databases(conn):
    q = """
    SELECT datname
    FROM pg_database
    WHERE datistemplate = false
      AND datallowconn = true
    ORDER BY datname;
    """
    with dict_cursor(conn) as cur:
        cur.execute(q)
        return [r["datname"] for r in cur.fetchall()]

def find_candidate_tables(conn):
    # Pull user tables plus sizes and row estimates
    q = """
    WITH sizes AS (
      SELECT
        c.oid,
        n.nspname AS schema,
        c.relname AS name,
        pg_total_relation_size(c.oid) AS total_bytes,
        pg_relation_size(c.oid) AS heap_bytes
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE c.relkind IN ('r','p') -- table or partitioned table
        AND n.nspname NOT IN ('pg_catalog','information_schema')
    )
    SELECT
      s.schema,
      s.name,
      s.total_bytes,
      s.heap_bytes,
      COALESCE(t.reltuples, 0) AS reltuples_est
    FROM sizes s
    LEFT JOIN pg_class t ON t.oid = s.oid
    ORDER BY s.total_bytes DESC NULLS LAST
    LIMIT 200; -- keep it reasonable
    """
    with dict_cursor(conn) as cur:
        cur.execute(q)
        return cur.fetchall()

def get_columns_for_table(conn, schema, table):
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """
    with dict_cursor(conn) as cur:
        cur.execute(q, (schema, table))
        return [r["column_name"] for r in cur.fetchall()]

def score_table(name, columns):
    score = 0
    name_l = name.lower()
    # Name-based hints
    for hint in NAME_HINTS:
        if hint in name_l:
            score += 2
    # Column-based hints
    cols_l = [c.lower() for c in columns]
    for hint in COLUMN_HINTS:
        if hint in cols_l:
            score += 1
    # Common OHLC quartet bonus
    ohlc_set = {"open", "high", "low", "close"}
    if ohlc_set.issubset(set(cols_l)):
        score += 3
    # Ticker/symbol + timestamp + price-ish bonuses
    if any(c in cols_l for c in ("ticker", "symbol", "instrument")):
        score += 1
    if any(c in cols_l for c in ("time", "timestamp", "ts", "datetime", "date")):
        score += 1
    if any(c in cols_l for c in ("price", "last", "mid", "bid", "ask", "open", "high", "low", "close")):
        score += 1
    return score

def pretty_size(num):
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}EB"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-port", type=int, default=5432)
    parser.add_argument("--max-port", type=int, default=5450)
    parser.add_argument("--hosts", type=str, default=",".join([h for h in SOCKET_DIRS if h is not None]))
    parser.add_argument("--include-socket-default", action="store_true",
                        help="Also try host=None (default unix socket search).")
    parser.add_argument("--min-score", type=int, default=2, help="Only show tables with score >= this.")
    parser.add_argument("--limit-dbs", type=int, default=0, help="Limit number of databases scanned per server (0 = no limit).")
    args = parser.parse_args()

    host_list = args.hosts.split(",") if args.hosts else []
    if args.include_socket_default:
        host_list = [None] + host_list

    # Build port list
    ports = list(range(args.min_port, args.max_port + 1))

    user = os.getenv("PGUSER") or getpass.getuser()

    print(f"# Searching for PostgreSQL servers as user '{user}'")
    print(f"# Hosts: {host_list}  Ports: {ports[0]}..{ports[-1]}")
    print()

    found_any = False

    for host in host_list or [None]:
        # For TCP, cheaply skip unopened ports
        candidate_ports = []
        if host in (None, "/var/run/postgresql", "/tmp"):
            candidate_ports = ports  # can't cheaply probe Unix sockets; just try
        else:
            for p in ports:
                if port_open(host, p):
                    candidate_ports.append(p)

        if not candidate_ports:
            continue

        for port in candidate_ports:
            ok, info = try_connect(host, port, user)
            if not ok:
                # Uncomment to see failures:
                # print(f"  [-] Cannot connect to host={host} port={port}: {info}")
                continue

            found_any = True
            host_disp = host if host is not None else "(unix-socket-default)"
            print(f"✅ Connected: host={host_disp} port={port}  |  {info}")

            # List DBs
            with connect_any({"host": host, "port": port, "user": user, "dbname": "postgres"}) as conn0:
                dbs = list_databases(conn0)

            if not dbs:
                print("   (No non-template databases)")
                continue

            if args.limit_dbs > 0:
                dbs = dbs[:args.limit_dbs]

            for db in dbs:
                print(f"\n  🔎 Scanning DB: {db}")
                try:
                    with connect_any({"host": host, "port": port, "user": user, "dbname": db}) as conn:
                        tables = find_candidate_tables(conn)
                        if not tables:
                            print("    (No user tables found)")
                            continue

                        ranked = []
                        for t in tables:
                            schema = t["schema"]
                            name = t["name"]
                            try:
                                cols = get_columns_for_table(conn, schema, name)
                            except Exception:
                                cols = []
                            s = score_table(name, cols)
                            ranked.append({
                                "schema": schema,
                                "table": name,
                                "score": s,
                                "size_bytes": int(t["total_bytes"] or 0),
                                "rows_est": int(t["reltuples_est"] or 0),
                                "columns": cols,
                            })

                        # Sort by score desc, then size desc
                        ranked.sort(key=lambda r: (r["score"], r["size_bytes"]), reverse=True)

                        shown_any = False
                        for r in ranked:
                            if r["score"] < args.min_score:
                                continue
                            shown_any = True
                            print(f"    • {db}.{r['schema']}.{r['table']}  "
                                  f"[score={r['score']}, size~{pretty_size(r['size_bytes'])}, rows_est~{r['rows_est']}]")
                            if r["columns"]:
                                # Show a short column preview
                                preview = ", ".join(r["columns"][:12])
                                more = "" if len(r["columns"]) <= 12 else " …"
                                print(f"      cols: {preview}{more}")

                        if not shown_any:
                            top = ranked[:5]
                            if top:
                                print("    (No strong matches; top tables by heuristic):")
                                for r in top:
                                    print(f"      - {db}.{r['schema']}.{r['table']} "
                                          f"[score={r['score']}, size~{pretty_size(r['size_bytes'])}]")

                except Exception as e:
                    print(f"    (Error scanning DB '{db}'): {e}")

            print()  # spacing between servers

    if not found_any:
        print("No reachable local PostgreSQL servers were found with the given hosts/ports and current user.")
        print("Tips:")
        print("  • If your server runs under a different system user, try: PGUSER=thatuser python find_pg_market_db.py")
        print("  • If it needs a password, store it in ~/.pgpass or set PGPASSWORD=<pw> in the env.")
        print("  • Try enabling default socket search: --include-socket-default")
        print("  • Broaden the port range: --max-port 5500")
        print("  • If Postgres is inside Docker, add the container’s host/port to --hosts/--min/--max-port.")
        print("  • On Debian/Ubuntu, domain sockets are usually in /var/run/postgresql; on others, /tmp.")
        print()

if __name__ == "__main__":
    main()
