#!/usr/bin/env python3
"""
pytorch_nightly_sizes.py

Fetch sizes of PyTorch NIGHTLY CUDA wheels by month.

- Picks the latest nightly per calendar month (based on the YYYYMMDD in the version, e.g. 2.x.dev20251107)
- Defaults: CUDA cu124, any Python ABI, linux_x86_64 wheels
- Outputs a neat table and CSV

Usage:
  python pytorch_nightly_sizes.py \
    --months 12 \
    --cuda cu124 \
    --platform linux_x86_64 \
    --py-abi ''   # e.g. cp311 (optional); empty means any

You can point to a different base URL with --base-url if needed.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_BASE_URL = "https://download.pytorch.org/whl/nightly/torch/"

WHEEL_RE = re.compile(
    r"""
    ^torch-                                   # package name
    (?P<version>[\d\.]+(?:\.dev(?P<date>\d{8}))?)  # version with optional .devYYYYMMDD
    (?:\+[^-]+)?-                             # optional local version +... (rare)
    (?P<pyabi>cp\d{2,3}|py3?|cp\d{2,3}m)?-    # python abi (optional; be lenient)
    (?P<tag1>[^-]+)-(?P<tag2>[^-]+)\.whl$     # platform tags (e.g., linux_x86_64)
    """,
    re.VERBOSE,
)

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def iter_months_back(months: int, start: datetime) -> list[datetime]:
    # returns a list of datetimes representing the first day of each month, newest first
    y, m = start.year, start.month
    out = []
    for _ in range(months):
        out.append(datetime(y, m, 1, tzinfo=timezone.utc))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return out

def parse_args():
    p = argparse.ArgumentParser(description="Get PyTorch nightly CUDA wheel sizes by month.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Index URL to scrape.")
    p.add_argument("--months", type=int, default=12, help="How many months back (including current).")
    p.add_argument("--cuda", default="cu124", help="CUDA tag to match, e.g. cu124, cu121.")
    p.add_argument("--platform", default="linux_x86_64", help="Platform tag to match (e.g., linux_x86_64).")
    p.add_argument("--py-abi", default="", help="Optional Python ABI to filter (e.g., cp311). Leave empty for any.")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds.")
    p.add_argument("--csv", default="pytorch_nightly_sizes.csv", help="Output CSV path.")
    return p.parse_args()

def fetch_index(base_url: str, timeout: float) -> str:
    r = requests.get(base_url, timeout=timeout)
    r.raise_for_status()
    return r.text

def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".whl"):
            links.append(href)
    return links

def parse_wheel(href: str):
    # href might be filename or full path; use basename
    name = href.split("/")[-1]
    m = WHEEL_RE.match(name)
    if not m:
        return None
    gd = m.groupdict()
    date_str = gd.get("date")
    dt = None
    if date_str:
        try:
            dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            dt = None
    pyabi = gd.get("pyabi") or ""
    platform = f"{gd.get('tag1','')}"
    if gd.get("tag2"):
        platform += f"_{gd['tag2']}"
    return {
        "name": name,
        "href": href,
        "version": gd.get("version") or "",
        "date": dt,  # may be None if no dev date present
        "pyabi": pyabi,
        "platform": platform,
    }

def head_size(url: str, timeout: float) -> int | None:
    # Try HEAD first; some servers may not return content-length on HEAD
    try:
        h = requests.head(url, allow_redirects=True, timeout=timeout)
        if h.ok:
            clen = h.headers.get("Content-Length")
            if clen and clen.isdigit():
                return int(clen)
    except requests.RequestException:
        pass
    # Fallback: GET with stream, but do not download body
    try:
        g = requests.get(url, stream=True, timeout=timeout)
        g.close()
        clen = g.headers.get("Content-Length")
        if clen and clen.isdigit():
            return int(clen)
    except requests.RequestException:
        pass
    return None

def main():
    args = parse_args()
    now = datetime.now(timezone.utc)

    try:
        html = fetch_index(args.base_url, args.timeout)
    except Exception as e:
        print(f"ERROR: failed to fetch index: {e}", file=sys.stderr)
        sys.exit(2)

    raw_links = extract_links(html)
    wheels = []
    for href in raw_links:
        info = parse_wheel(href)
        if not info:
            continue
        wheels.append(info)

    # Filters
    filtered = []
    for w in wheels:
        # CUDA tag is typically embedded in the filename for torch wheels (e.g., +cu124 or -cu124-)
        # We match conservatively against the filename.
        name = w["name"]
        if args.cuda not in name:
            continue
        if args.platform and not name.endswith(f"{args.platform}.whl"):
            continue
        if args.py_abi and args.py_abi not in name:
            continue
        # Prefer wheels with a dev date so we can bucket by month
        filtered.append(w)

    if not filtered:
        print("No matching wheels found with the given filters.", file=sys.stderr)
        sys.exit(3)

    # Group by month and pick the latest nightly within each month
    # If a wheel lacks a date, we skip it for monthly selection.
    by_month = defaultdict(list)
    for w in filtered:
        if w["date"] is None:
            continue
        mk = month_key(w["date"])
        by_month[mk].append(w)

    target_months = [month_key(d) for d in iter_months_back(args.months, now)]

    chosen = []
    for mk in target_months:
        candidates = by_month.get(mk, [])
        if not candidates:
            chosen.append({"month": mk, "name": None, "url": None, "size": None, "date": None, "pyabi": None})
            continue
        # pick latest by date
        best = max(candidates, key=lambda x: x["date"])
        url = best["href"]
        if not url.startswith("http"):
            url = urljoin(args.base_url, url)
        size = head_size(url, args.timeout)
        chosen.append({
            "month": mk,
            "name": best["name"],
            "url": url,
            "size": size,
            "date": best["date"].strftime("%Y-%m-%d"),
            "pyabi": best["pyabi"],
        })

    # Print table
    print(f"\nPyTorch NIGHTLY wheel sizes (CUDA={args.cuda}, platform={args.platform}, py_abi={'ANY' if not args.py_abi else args.py_abi})")
    print(f"Source index: {args.base_url}\n")
    print(f"{'Month':<8}  {'Date':<10}  {'Size (MB)':>10}  {'PyABI':<7}  {'Wheel'}")
    print("-" * 100)
    for row in chosen:
        sz_mb = f"{row['size']/1_000_000:.2f}" if row["size"] is not None else "NA"
        print(f"{row['month']:<8}  {str(row['date'] or 'NA'):<10}  {sz_mb:>10}  {str(row['pyabi'] or 'NA'):<7}  {row['name'] or 'No wheel found'}")

    # Write CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "date", "size_bytes", "size_mb", "pyabi", "wheel_name", "url"])
        for row in chosen:
            size_b = row["size"] if row["size"] is not None else ""
            size_mb = (row["size"]/1_000_000) if row["size"] is not None else ""
            writer.writerow([
                row["month"],
                row["date"] or "",
                size_b,
                size_mb,
                row["pyabi"] or "",
                row["name"] or "",
                row["url"] or "",
            ])
    print(f"\nCSV written to: {args.csv}")

if __name__ == "__main__":
    main()

