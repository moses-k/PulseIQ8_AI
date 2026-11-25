"""How-to knowledge for the support assistant: how AnalyticsIQ works and how to use it.

This is product guidance only (no sales data), so the assistant can answer "how do I…" and "why does…" questions
without calling AnalyticsIQ. Keep it in step with docs/AnalyticsIQ_Functional_Specification.docx.
"""
from __future__ import annotations

import re

TOPICS: dict[str, dict] = {
    "upload": {
        "title": "Uploading a sales file",
        "keywords": "upload excel file import sheet xlsx replace reupload duplicate skipped rows",
        "text": """Open **Upload Excel** (or the *Upload sales file* button), choose the .xlsx sales export and upload it.
- The first sheet with the sales columns is read (date, document, customer, rep, region, segment, amount).
- Rows without a valid date, customer or amount, and totals rows, are skipped; the upload result shows how many.
- Uploading a file that covers dates already loaded REPLACES those dates (no double counting); the result shows replaced rows.
- Claim lines ("<DISTRIBUTOR> CLAIMS", e.g. TRANSWIDE CLAIMS) stay in sales and are credited to the claiming rep and region.
- New customers are created automatically and listed under Customers → *To review* so you can confirm segment, region and distributor.
- Only administrators can upload or delete uploads.""",
    },
    "review": {
        "title": "Customers to review",
        "keywords": "review new customer customers to review onboard segment region assign master unassigned",
        "text": """Customers seen for the first time in a sales file are marked *to review* (Customers page, filter *To review*).
Check each one: segment, region and, for distributor customers, the distributor. A customer may belong to more than one
segment or region (e.g. ADAKIM CHEMIST - INSTITUTIONS and ADAKIM CHEMIST - SOUTH RIFT): sales follow the segment and region on
each sales line. Lines with no rep, region or segment show as UNASSIGNED until the master data is fixed.""",
    },
    "attribution": {
        "title": "How sales are credited (direct, distributor, claims)",
        "keywords": "credited direct distributor indirect attribution region rep counted why total different company total",
        "text": """- **Company total** = every invoiced line.
- **Direct sales** count for the customer's region and rep.
- **Distributor purchases** (JACARANDA, MEDS, KENTONS LIMITED, TRANSWIDE …) are company revenue but count for no rep or region until claimed.
- A distributor such as KENTONS may also have branches bought direct; those branches count as direct sales.
- **Claims**: a rep claims sales a distributor made to a customer that is not onboarded. Claim lines in sales files and
approved claims count for the claiming rep and region, shown as *claimed*.
- So region / rep totals (direct + claims) can be lower than the company total; the difference is unclaimed distributor purchases.""",
    },
    "claims": {
        "title": "Distributor claims",
        "keywords": "claim claims distributor transwide approve reject pending unclaimed",
        "text": """Open **Distributor claims** to see claims and their status (PENDING, APPROVED, REJECTED).
Claim lines inside the sales files ("TRANSWIDE CLAIMS") are counted automatically and credited to the rep and region on the line.
Only approved claims move sales to a rep/region. The AI assistant can list claims and unclaimed purchases but cannot approve,
reject or edit claims — an administrator does that on the claims page.""",
    },
    "targets": {
        "title": "Targets, achievement and projection",
        "keywords": "target targets achievement pace projection month end behind ahead set targets tiers performer risk critical",
        "text": """Targets are monthly, by segment, region or rep (**Targets** page; administrators edit them).
- Achievement = sales ÷ target.
- Month elapsed = days elapsed ÷ days in the month (days of the latest data).
- Pace ratio = achievement ÷ month elapsed: Performers ≥ 0.9, Mid performers 0.7–0.9, At risk 0.45–0.7, Critical < 0.45.
- Projection = sales so far ÷ days elapsed × days in the month.""",
    },
    "regions": {
        "title": "Regions and region names in files",
        "keywords": "region regions zone mapping alias unmapped name file",
        "text": """**Regions** lists the sales regions and the names used for them in sales files (aliases). A file region name that maps
to no region is reported as *unmapped*; add it as an alias of the right region and the sales are counted there.
Reps belong to a region; a rep without a region is listed as a data issue.""",
    },
    "reports": {
        "title": "Reports and exports",
        "keywords": "report reports export csv download summary management performance daily",
        "text": """- **Dashboard**: KPIs, performance overview, target, sales lines.
- **Performance reports**: rankings by rep, region and segment.
- **Management summary**: the month in one page for management.
- **Sales data**: every line with filters, exportable to CSV.
Use the filters (period, region, segment, rep) at the top; the assistant uses the same filters when you ask from a page.""",
    },
    "access": {
        "title": "Sign-in, roles and WhatsApp access",
        "keywords": "login sign in password role admin manager rep access whatsapp phone register user users permission",
        "text": """Everyone signs in with a username and password; change the temporary password at first sign-in.
- **ADMIN**: everything, including uploads, master data, targets, users and the AI audit.
- **MANAGER**: reads everything (or only their region when a region is set); cannot change data.
- **REP**: sees only their own sales, customers and targets.
WhatsApp: an administrator adds your phone number to your user (Nexora AI → Users & access) and enables WhatsApp.
The WhatsApp assistant answers with exactly the same permissions as your web sign-in.""",
    },
    "assistant": {
        "title": "What the AI assistant can do",
        "keywords": "ai assistant nexora what can you do help commands capabilities whatsapp brief",
        "text": """Ask in plain language, for example:
- "How are we doing this month?" · "Which region is behind target?" · "Rank reps by achievement"
- "Will we hit the target?" · "Which customers stopped buying?" · "Which customers are declining?"
- "How much did ADAKIM buy this month?" · "Unclaimed TRANSWIDE sales" · "Give me the management briefing"
The assistant only reads data through approved AnalyticsIQ tools, within your role. It cannot approve claims, change targets,
edit master data or delete uploads. On WhatsApp send *help*, *brief* (briefing) or *reset* (new conversation).""",
    },
}


def search(query: str, limit: int = 3) -> list[dict]:
    words = [w for w in re.findall(r"[a-z]+", (query or "").lower()) if len(w) > 2]
    scored = []
    for key, t in TOPICS.items():
        hay = f"{key} {t['title']} {t['keywords']}".lower()
        score = sum(2 if w in t["keywords"].split() else 1 for w in words if w in hay)
        if score:
            scored.append((score, key))
    scored.sort(reverse=True)
    keys = [k for _, k in scored[:limit]] or ["assistant"]
    return [{"topic": k, "title": TOPICS[k]["title"], "guide": TOPICS[k]["text"]} for k in keys]
