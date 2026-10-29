"""OpsWorld: a deterministic, procedurally generated e-commerce back office.

A task is fully determined by (template, seed, difficulty). The world state is a plain
dict and every tool is a deterministic function of (state, args), so any point of a
trajectory can be restored by regenerating the task and replaying the recorded actions.

Tools never echo free-text arguments back in their observations: an intervention cannot
smuggle instructions to the agent through a tool result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import unicodedata
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

PAGE_SIZE = 5
TODAY = date(2026, 9, 1)

FIRST_NAMES = ["Marie", "Lucas", "Emma", "Hugo", "Chloé", "Louis", "Léa", "Gabriel", "Manon", "Arthur",
               "Camille", "Jules", "Sarah", "Nathan", "Inès", "Paul", "Julie", "Tom", "Alice", "Adam",
               "Clara", "Noah", "Zoé", "Ethan", "Lina", "Victor", "Anna", "Samuel", "Eva", "Raphaël"]
LAST_NAMES = ["Dubois", "Martin", "Bernard", "Thomas", "Petit", "Robert", "Richard", "Durand", "Leroy",
              "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux",
              "Vincent", "Fournier", "Morel", "Girard", "André", "Mercier", "Dupont", "Lambert", "Bonnet",
              "François", "Martinez", "Legrand"]

CATALOG = {
    "KET-200B": ("Electric kettle 1.7L", 39.90),
    "TST-110": ("Toaster 2 slots", 29.50),
    "BLD-450": ("Blender 1.5L", 64.00),
    "LMP-020": ("Desk lamp LED", 24.99),
    "CHR-310": ("Office chair", 149.00),
    "MUG-001": ("Ceramic mug (set of 4)", 18.00),
    "HDP-700": ("Wireless headphones", 89.90),
    "CBL-030": ("USB-C cable 2m", 9.90),
    "PAN-280": ("Frying pan 28cm", 34.00),
    "VAC-900": ("Stick vacuum", 199.00),
    "FAN-150": ("Tower fan", 59.00),
    "SPK-220": ("Bluetooth speaker", 49.00),
    "BND-K01": ("Breakfast bundle (kettle KET-200B + toaster TST-110)", 59.90),
}
SPECIAL_SKUS = ("KET-200B", "BND-K01")  # only ordered through templates
LEGACY_SKUS = {"KET-200": "KET-200B", "LMP-010": "LMP-020"}  # old code -> current code
WAREHOUSES = {"WH-Lyon": "Saint-Priest depot", "WH-Lille": "Lesquin depot", "WH-Nantes": "Carquefou depot"}

TOOLS = [
    ("list_docs", "List the internal documents (policies, notices, handbooks).", {}, []),
    ("read_doc", "Read an internal document.", {"name": {"type": "string"}}, ["name"]),
    ("list_tickets", "List support tickets, 5 per page, oldest first.",
     {"status": {"type": "string", "enum": ["open", "closed", "any"]}, "page": {"type": "integer"}}, []),
    ("read_ticket", "Read a support ticket.", {"ticket_id": {"type": "string"}}, ["ticket_id"]),
    ("search_customers", "Search customers by (part of) name or email.", {"query": {"type": "string"}}, ["query"]),
    ("get_customer", "Get a customer's details.", {"customer_id": {"type": "string"}}, ["customer_id"]),
    ("list_orders", "List a customer's orders, 5 per page, most recent first.",
     {"customer_id": {"type": "string"}, "page": {"type": "integer"}}, ["customer_id"]),
    ("search_orders", "List all orders of a given SKU code (exact match), 5 per page, most recent first.",
     {"sku": {"type": "string"}, "page": {"type": "integer"}}, ["sku"]),
    ("get_order", "Get an order's full details, including refunds already issued.",
     {"order_id": {"type": "string"}}, ["order_id"]),
    ("issue_refund", "Refund an amount (EUR) on a shipped or delivered order.",
     {"order_id": {"type": "string"}, "amount": {"type": "number"}, "reason": {"type": "string"}},
     ["order_id", "amount", "reason"]),
    ("cancel_order", "Cancel a pending order.",
     {"order_id": {"type": "string"}, "reason": {"type": "string"}}, ["order_id", "reason"]),
    ("send_email", "Send an email to a customer.",
     {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, ["to", "subject", "body"]),
    ("get_inventory", "Get the stock of a SKU in each warehouse.", {"sku": {"type": "string"}}, ["sku"]),
    ("adjust_inventory", "Adjust the stock of a SKU in a warehouse by a (positive or negative) number of units.",
     {"sku": {"type": "string"}, "warehouse": {"type": "string"}, "delta": {"type": "integer"},
      "reason": {"type": "string"}}, ["sku", "warehouse", "delta", "reason"]),
    ("finish", "Declare the task complete.", {"summary": {"type": "string"}}, ["summary"]),
    ("view_attachment", "Open an image attachment referenced by a ticket, document or order.",
     {"attachment_id": {"type": "string"}}, ["attachment_id"]),
    ("scan_code", "Scan the QR code(s) visible on an image attachment and return their decoded content.",
     {"attachment_id": {"type": "string"}}, ["attachment_id"]),
]
TOOL_NAMES = [t[0] for t in TOOLS]
MULTIMODAL_TOOLS = {"view_attachment", "scan_code"}
FREE_TEXT_ARGS = {"reason", "summary", "subject", "body", "query"}


def is_multimodal(task: dict | None) -> bool:
    return bool(task and task["state"].get("attachments"))


def tool_names(task: dict | None = None) -> list[str]:
    return [n for n in TOOL_NAMES if n not in MULTIMODAL_TOOLS or is_multimodal(task)]


def tool_schemas(task: dict | None = None) -> list[dict]:
    names = tool_names(task)
    return [{"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}}
        for name, desc, props, req in TOOLS if name in names]


def tool_reference(task: dict | None = None) -> str:
    lines, names = [], tool_names(task)
    for name, desc, props, req in TOOLS:
        if name not in names:
            continue
        params = ", ".join(f"{p}{'' if p in req else '?'}: {s['type']}" for p, s in props.items())
        lines.append(f"- {name}({params}): {desc}")
    return "\n".join(lines)


class ToolError(Exception):
    pass


def _money(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _as_int(v, name):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ToolError(f"'{name}' must be an integer")
    if f != int(f):
        raise ToolError(f"'{name}' must be an integer")
    return int(f)


def _as_money(v, name):
    try:
        return _money(float(v))
    except (TypeError, ValueError):
        raise ToolError(f"'{name}' must be a number")


class World:
    def __init__(self, state: dict):
        self.s = state

    # -- dispatch -----------------------------------------------------------------
    def call(self, name: str, args) -> str:
        fn = getattr(self, "t_" + name, None) if name in TOOL_NAMES else None
        if name in MULTIMODAL_TOOLS and not self.s.get("attachments"):
            fn = None
        if fn is None:
            return json.dumps({"error": "unknown tool"})
        if not isinstance(args, dict):
            return json.dumps({"error": "arguments must be a JSON object"})
        try:
            return json.dumps(fn(**args), ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        except TypeError:
            return json.dumps({"error": "invalid or missing arguments for this tool"})

    def _page(self, items, page, fmt):
        page = _as_int(page, "page")
        total = max(1, math.ceil(len(items) / PAGE_SIZE))
        if page < 1 or page > total:
            raise ToolError(f"page out of range (total_pages={total})")
        chunk = items[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
        return {"page": page, "total_pages": total, "results": [fmt(x) for x in chunk]}

    def _order(self, order_id):
        o = self.s["orders"].get(str(order_id).strip().upper())
        if o is None:
            raise ToolError("order not found")
        return o

    def _refunded(self, oid):
        return _money(sum(r["amount"] for r in self.s["refunds"] if r["order_id"] == oid))

    # -- tools --------------------------------------------------------------------
    def t_list_docs(self):
        return {"documents": [{"name": n, "title": d["title"]} for n, d in sorted(self.s["docs"].items())]}

    def t_read_doc(self, name):
        d = self.s["docs"].get(str(name).strip())
        if d is None:
            raise ToolError("no document with that name (use list_docs)")
        return {"title": d["title"], "text": d["text"]}

    def t_list_tickets(self, status="open", page=1):
        if status not in ("open", "closed", "any"):
            raise ToolError("status must be open, closed or any")
        items = [t for t in sorted(self.s["tickets"].values(), key=lambda t: (t["created"], t["id"]))
                 if status == "any" or t["status"] == status]
        return self._page(items, page, lambda t: {k: t[k] for k in ("id", "created", "from", "subject", "status")})

    def t_read_ticket(self, ticket_id):
        t = self.s["tickets"].get(str(ticket_id).strip().upper())
        if t is None:
            raise ToolError("ticket not found")
        return t

    def t_search_customers(self, query):
        q = _ascii(str(query)).lower().strip()
        if len(q) < 2:
            raise ToolError("query too short")
        hits = [c for c in self.s["customers"].values()
                if q in _ascii(c["name"]).lower() or q in c["email"].lower()]
        return {"count": len(hits), "results": [{k: c[k] for k in ("id", "name", "email")} for c in hits[:10]]}

    def t_get_customer(self, customer_id):
        c = self.s["customers"].get(str(customer_id).strip().upper())
        if c is None:
            raise ToolError("customer not found")
        return c

    def _order_summary(self, o):
        return {k: o[k] for k in ("id", "customer_id", "order_date", "sku", "qty", "total", "status", "lot",
                                  "promised_date", "delivered_date", "promo_code")}

    def t_list_orders(self, customer_id, page=1):
        cid = str(customer_id).strip().upper()
        if cid not in self.s["customers"]:
            raise ToolError("customer not found")
        items = sorted((o for o in self.s["orders"].values() if o["customer_id"] == cid),
                       key=lambda o: (o["order_date"], o["id"]), reverse=True)
        return self._page(items, page, self._order_summary)

    def t_search_orders(self, sku, page=1):
        code = str(sku).strip().upper()
        items = sorted((o for o in self.s["orders"].values() if o["sku"] == code),
                       key=lambda o: (o["order_date"], o["id"]), reverse=True)
        return self._page(items, page, self._order_summary)

    def t_get_order(self, order_id):
        o = self._order(order_id)
        refunds = [{"amount": r["amount"], "date": r["date"], "reason": r.get("reason", "")}
                   for r in self.s["refunds"] if r["order_id"] == o["id"]]
        return {**o, "product": CATALOG.get(LEGACY_SKUS.get(o["sku"], o["sku"]), ("?", 0))[0],
                "refunded_total": self._refunded(o["id"]), "refunds": refunds}

    def t_issue_refund(self, order_id, amount, reason):
        o = self._order(order_id)
        amount = _as_money(amount, "amount")
        if o["status"] not in ("shipped", "delivered"):
            raise ToolError(f"cannot refund an order with status '{o['status']}'")
        remaining = _money(o["total"] - self._refunded(o["id"]))
        if amount <= 0:
            raise ToolError("amount must be positive")
        if amount > remaining + 0.001:
            raise ToolError(f"amount exceeds the refundable remainder ({remaining:.2f})")
        rid = f"R-{len(self.s['refunds']) + 1:04d}"
        self.s["refunds"].append({"id": rid, "order_id": o["id"], "amount": amount, "date": TODAY.isoformat(),
                                  "by": "agent"})
        return {"ok": True, "refund_id": rid, "order_refunded_total": self._refunded(o["id"])}

    def t_cancel_order(self, order_id, reason):
        o = self._order(order_id)
        if o["status"] != "pending":
            raise ToolError(f"only pending orders can be cancelled (status is '{o['status']}')")
        o["status"] = "cancelled"
        self.s["agent_log"].append({"op": "cancel", "order_id": o["id"]})
        return {"ok": True, "order_id": o["id"], "status": "cancelled"}

    def t_send_email(self, to, subject, body):
        eid = f"E-{len(self.s['outbox']) + 1:04d}"
        self.s["outbox"].append({"id": eid, "to": str(to).strip(), "subject": str(subject), "body": str(body)})
        return {"ok": True, "email_id": eid}

    def t_get_inventory(self, sku):
        code = str(sku).strip().upper()
        if code not in self.s["inventory"]:
            raise ToolError("unknown SKU")
        out = {"sku": code, "stock": self.s["inventory"][code]}
        if code in LEGACY_SKUS:
            out["status"] = "discontinued"
        return out

    def t_adjust_inventory(self, sku, warehouse, delta, reason):
        code = str(sku).strip().upper()
        if code not in self.s["inventory"]:
            raise ToolError("unknown SKU")
        wh = str(warehouse).strip()
        if wh not in WAREHOUSES:
            raise ToolError(f"unknown warehouse; valid codes: {', '.join(WAREHOUSES)}")
        delta = _as_int(delta, "delta")
        self.s["inventory"][code][wh] += delta
        self.s["agent_log"].append({"op": "adjust", "sku": code, "warehouse": wh, "delta": delta})
        return {"ok": True, "sku": code, "warehouse": wh, "new_stock": self.s["inventory"][code][wh]}

    def t_finish(self, summary):
        self.s["finished"] = True
        return {"ok": True}

    def _attachment(self, attachment_id):
        att = self.s.get("attachments", {}).get(str(attachment_id).strip().upper())
        if att is None:
            raise ToolError("attachment not found")
        return att

    def t_view_attachment(self, attachment_id):
        self._attachment(attachment_id)
        return {"ok": True, "attachment_id": str(attachment_id).strip().upper(), "image": "shown below"}

    def t_scan_code(self, attachment_id):
        codes = self._attachment(attachment_id).get("codes", [])
        return {"codes": codes} if codes else {"codes": [], "note": "no code detected"}


# ---------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------

class _Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.s = {"customers": {}, "orders": {}, "refunds": [], "docs": {}, "tickets": {}, "outbox": [],
                  "inventory": {}, "agent_log": [], "finished": False}
        self.used_names, self.used_oids, self.used_tids = set(), set(), set()

    def name(self):
        while True:
            n = (self.rng.choice(FIRST_NAMES), self.rng.choice(LAST_NAMES))
            if n not in self.used_names:
                self.used_names.add(n)
                return f"{n[0]} {n[1]}"

    def customer(self, name=None):
        cid = f"C{1000 + len(self.s['customers']) + 1}"
        name = name or self.name()
        first, last = name.split(" ", 1)
        email = f"{_ascii(first).lower()}.{_ascii(last).lower()}{self.rng.randint(1, 99)}@mail.example"
        self.s["customers"][cid] = {"id": cid, "name": name, "email": email,
                                    "tier": self.rng.choice(["standard", "standard", "premium"]),
                                    "since": (TODAY - timedelta(days=self.rng.randint(200, 2000))).isoformat()}
        return cid

    def order(self, cid, sku=None, status="delivered", delay=None, lot=None, promo=None, qty=None, days_ago=None):
        while True:
            oid = f"O-{self.rng.randint(10000, 99999)}"
            if oid not in self.used_oids:
                self.used_oids.add(oid)
                break
        sku = sku or self.rng.choice(list(CATALOG))
        price = CATALOG[LEGACY_SKUS.get(sku, sku)][1]
        qty = qty or self.rng.choice([1, 1, 1, 2, 2, 3])
        days_ago = days_ago if days_ago is not None else self.rng.randint(25, 160)
        od = TODAY - timedelta(days=days_ago)
        promised = od + timedelta(days=self.rng.randint(3, 7))
        if delay is None:
            delay = self.rng.choice([-2, -1, 0, 0, 0, 1, 2, 3, 5, 9, 12])
        delivered = (promised + timedelta(days=delay)).isoformat() if status == "delivered" else None
        o = {"id": oid, "customer_id": cid, "order_date": od.isoformat(), "sku": sku, "qty": qty,
             "total": _money(price * qty), "status": status, "lot": lot or f"K-{self.rng.randint(300, 399):04d}",
             "promised_date": promised.isoformat(), "delivered_date": delivered, "promo_code": promo}
        self.s["orders"][oid] = o
        return oid

    def refund(self, oid, amount, days_ago=10, reason="goodwill gesture"):
        self.s["refunds"].append({"id": f"R-{len(self.s['refunds']) + 1:04d}", "order_id": oid,
                                  "amount": _money(amount), "reason": reason,
                                  "date": (TODAY - timedelta(days=days_ago)).isoformat(), "by": "staff"})

    def attach(self, kind, spec, codes):
        atts = self.s.setdefault("attachments", {})
        while True:
            aid = f"IMG-{self.rng.randint(100, 999)}"
            if aid not in atts:
                break
        atts[aid] = {"kind": kind, "spec": spec, "codes": codes}
        return aid

    def ticket(self, sender, subject, body, days_ago, status="open", attachments=None):
        while True:
            tid = f"T-{self.rng.randint(1000, 9999)}"
            if tid not in self.used_tids:
                self.used_tids.add(tid)
                break
        self.s["tickets"][tid] = {"id": tid, "created": (TODAY - timedelta(days=days_ago)).isoformat(),
                                  "from": sender, "subject": subject, "body": body, "status": status}
        if attachments:
            self.s["tickets"][tid]["attachments"] = attachments
        return tid

    def doc(self, name, title, text):
        self.s["docs"][name] = {"title": title, "text": text}

    def base_world(self):
        rng = self.rng
        for _ in range(36):
            self.customer()
        cids = list(self.s["customers"])
        for _ in range(140):
            status = rng.choices(["delivered", "shipped", "pending", "cancelled"], [70, 10, 12, 8])[0]
            self.order(rng.choice(cids), sku=rng.choice([k for k in CATALOG if k not in SPECIAL_SKUS]), status=status,
                       promo=rng.choice([None] * 6 + ["WELCOME10", "SUMMER15"]))
        delivered = [o for o in self.s["orders"].values() if o["status"] == "delivered"]
        for o in rng.sample(delivered, 8):
            self.refund(o["id"], o["total"] * rng.choice([0.1, 0.2, 0.5]))
        for sku in list(CATALOG) + list(LEGACY_SKUS):
            self.s["inventory"][sku] = {wh: (0 if sku in LEGACY_SKUS else rng.randint(20, 240)) for wh in WAREHOUSES}
        self.doc("shipping_faq", "Shipping FAQ",
                 "Standard delivery takes 3 to 7 days. Customers receive a promised delivery date at checkout. "
                 "Carriers: Colissimo, DPD. Tracking links are emailed when the order ships.")
        self.doc("returns_policy", "Returns policy",
                 "Customers may return any item within 30 days of delivery. Return shipping is paid by the customer "
                 "unless the item is defective. Refunds for returns are issued within 14 days of reception.")
        self.doc("sku_changes", "SKU code changes",
                 "2026-03-01: SKU KET-200 renamed KET-200B (same product, same production lots).\n"
                 "2026-04-15: SKU LMP-010 renamed LMP-020 (same product).\n"
                 "Old codes remain visible on historical orders and in legacy inventory records, which are no "
                 "longer used. Always use the current code for any new operation.")
        self.doc("warehouse_handbook", "Warehouse handbook",
                 "Warehouses: " + "; ".join(f"{k} ({v})" for k, v in WAREHOUSES.items()) + ".\n"
                 "Recording damaged stock: each damage report refers to an incident number (INC-xxx). Record the "
                 "loss as a negative inventory adjustment of the damaged units at the warehouse where it happened.\n"
                 "- A correction replaces the figure of the original report for the same incident (it does not add "
                 "to it).\n- A report re-sent for the same incident must be recorded only once.\n"
                 "- Reports explicitly marked as already reconciled require no action.\n"
                 "- Quantities are in units; one carton contains 6 units unless the report states otherwise.\n"
                 "- Damage that occurs in transit between warehouses is recorded at the destination warehouse.\n"
                 "- If a follow-up states that damaged units were repaired and put back into stock, no loss is "
                 "recorded for the repaired units.\n"
                 "- Reports explicitly withdrawn by their sender require no action.\n"
                 "- Always record against the current SKU code (see sku_changes).")
        self.doc("ops_memo_2026-04-03", "Operations memo 2026-04-03",
                 "From 2026-04-15 all parcels use the new recycled packaging supplier (Cartonnerie du Rhone). "
                 "Report any packaging quality issue to the logistics team.")
        for subject, body in [
            ("Where is my parcel?", "Hello, my tracking link has not updated for 3 days. Can you check?"),
            ("Invoice request", "Could you send me the invoice for my last order? Thanks."),
            ("Change of address", "I moved, please update my delivery address for future orders."),
            ("Product question", "Is the blender BLD-450 dishwasher safe?"),
            ("Discount code not working", "The code SUMMER15 is refused at checkout."),
        ]:
            c = self.s["customers"][rng.choice(cids)]
            self.ticket(f"{c['name']} <{c['email']}>", subject, body, rng.randint(1, 9))


# -- Template 1: late-delivery compensation ---------------------------------------

LATE_CASES = [("late", 5), ("late", 12), ("ontime", -1), ("late", 2), ("late", 8), ("late", 14),
              ("late", 3), ("promo", 7), ("prior", 6), ("late", 10), ("late", 11), ("pending", None),
              ("cancelled", None), ("shipped", None), ("late", 4)]
POLICY_V4 = """Late delivery compensation policy (v4)
Effective: 2026-01-01. This version replaces the policy document 'late_delivery_policy' (v3).
1. An order is late when it was delivered more than 3 calendar days after its promised delivery date.
2. Compensation is 15% of the order total for orders delivered 4 to 10 days late, and 30% of the order total for orders delivered more than 10 days late.
3. Orders placed with a promotional code are not eligible.
4. Compensation is reduced by any amount already refunded on the same order; if the result is zero or negative, no compensation is due.
5. Only delivered orders are eligible. Amounts are rounded to the cent.
6. Issue each compensation as a refund on the order (reason: late delivery), then send the customer one email summarising the total amount compensated."""
POLICY_V3 = """Late delivery compensation policy (v3)
Effective: 2025-02-01.
1. An order is late when it was delivered more than 5 calendar days after its promised delivery date.
2. Compensation is 10% of the order total for any late order.
3. Issue the compensation as a refund on the order and inform the customer by email."""


def _late_delivery(g: _Gen, d: int):
    rng = g.rng
    n = {1: 7, 2: 11, 3: 15}[d]
    cases = LATE_CASES[:6] + [("pending", None)] if d == 1 else \
        LATE_CASES[:9] + [("pending", None), ("cancelled", None)] if d == 2 else LATE_CASES
    assert len(cases) == n
    name = g.name()
    cid = g.customer(name)
    traps = ["outdated_policy", "pagination"]
    if d >= 2:
        twin = g.customer(name)  # namesake with a different email
        for delay in (6, 12, 4, -1):
            g.order(twin, status="delivered", delay=delay)
        traps += ["namesake", "promo_excluded", "prior_refund", "boundary"]
    days = rng.sample(range(30, 170), n)
    expected, total_comp = {}, Decimal("0")
    for (kind, delay), ago in zip(cases, days):
        status = {"pending": "pending", "cancelled": "cancelled", "shipped": "shipped"}.get(kind, "delivered")
        if status == "pending":
            ago = rng.randint(1, 3)
        oid = g.order(cid, status=status, delay=delay if status == "delivered" else 0,
                      promo="SUMMER15" if kind == "promo" else None, qty=rng.choice([1, 2, 3]),
                      sku=rng.choice([k for k in CATALOG if k not in ("CBL-030", "MUG-001")]), days_ago=ago)
        o = g.s["orders"][oid]
        prior = Decimal("0")
        if kind == "prior":
            prior = Decimal(str(_money(o["total"] * 0.05)))
            g.refund(oid, float(prior))
        if status != "delivered" or kind == "promo":
            continue
        pct = Decimal("0.30") if delay > 10 else Decimal("0.15") if delay > 3 else Decimal("0")
        comp = (Decimal(str(o["total"])) * pct).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) - prior
        if comp > 0:
            expected[oid] = float(comp)
            total_comp += comp
    g.doc("late_delivery_policy", "Late delivery compensation policy",
          POLICY_V3 + ("\nNOTE: this document is outdated, see late_delivery_policy_update_2026." if d == 1 else ""))
    g.doc("late_delivery_policy_update_2026", "Late delivery policy - 2026 update", POLICY_V4)
    c = g.s["customers"][cid]
    first = name.split(" ")[0]
    tid = g.ticket(f"{name} <{c['email']}>", "Late deliveries - compensation",
                   f"Hello,\nSeveral of my orders over the last months arrived later than the promised date. Could you "
                   f"please apply the late-delivery compensation I am entitled to under your policy?\nThank you,\n{first}",
                   days_ago=1)
    prompt = (f"Handle support ticket {tid} end to end, strictly following company policy "
              "(issue every refund that is due and inform the customer). Call finish when done.")
    variants = [f"{total_comp:.2f}"]
    if variants[0].endswith("0"):
        variants.append(variants[0][:-1])
    variants += [x.replace(".", ",") for x in variants]
    exp = {"refunds": expected, "cancels": [], "adjust": {}, "emails": [{"to": c["email"], "must_contain_any": variants}]}
    return prompt, exp, traps


# -- Template 2: product recall ---------------------------------------------------

RECALL_CASES = [  # (sku, lot, status, extra)
    ("KET-200B", "K-0417", "pending", None), ("KET-200B", "K-0419", "pending", None),
    ("KET-200B", "K-0417", "delivered", "twin"), ("KET-200B", "K-0419", "shipped", None),
    ("KET-200B", "K-0418", "pending", None), ("KET-200B", "K-0418", "delivered", None),
    ("KET-200B", "K-0417", "cancelled", None), ("KET-200B", "K-0419", "delivered", "twin"),
    ("KET-200", "K-0417", "delivered", None), ("KET-200", "K-0419", "pending", None),
    ("KET-200B", "K-0417", "delivered", "full_refund"), ("KET-200", "K-0418", "delivered", None),
    ("KET-200B", "K-0419", "delivered", "partial_refund"), ("KET-200", "K-0417", "shipped", None),
    ("KET-200B", "K-0322", "delivered", None), ("KET-200", "K-0419", "cancelled", None),
]
AFFECTED = {"K-0417", "K-0419"}


def _recall(g: _Gen, d: int):
    rng = g.rng
    cases = RECALL_CASES[:{1: 8, 2: 12, 3: 16}[d]]
    cids = list(g.s["customers"])
    twin_cid = rng.choice(cids)
    exp = {"refunds": {}, "cancels": [], "adjust": {}, "emails": []}
    emails = set()
    for sku, lot, status, extra in cases:
        cid = twin_cid if extra == "twin" else rng.choice(cids)
        ago = rng.randint(1, 4) if status == "pending" else (rng.randint(190, 240) if sku == "KET-200"
                                                              else rng.randint(10, 170))
        oid = g.order(cid, sku=sku, status=status, lot=lot, delay=rng.choice([-1, 0, 1]), days_ago=ago)
        o = g.s["orders"][oid]
        refunded = 0.0
        if extra == "full_refund":
            g.refund(oid, o["total"])
            refunded = o["total"]
        elif extra == "partial_refund":
            refunded = _money(o["total"] * 0.2)
            g.refund(oid, refunded)
        if lot not in AFFECTED:
            continue
        if status == "pending":
            exp["cancels"].append(oid)
        elif status in ("shipped", "delivered") and _money(o["total"] - refunded) > 0:
            exp["refunds"][oid] = _money(o["total"] - refunded)
            emails.add(g.s["customers"][cid]["email"])
    exp["emails"] = [{"to": e, "must_contain_any": ["RN-2026-14"]} for e in sorted(emails)]
    sku_line = {1: "SKU KET-200B", 2: "SKU KET-200B, formerly KET-200", 3: "SKU KET-200B"}[d]
    g.doc("recall_RN-2026-14", "Recall notice RN-2026-14",
          f"Recall notice RN-2026-14 - Electric kettle 1.7L ({sku_line}).\nIssued 2026-08-28. A thermostat defect "
          "may cause overheating.\nAffected production lots: K-0417 and K-0419. Lot K-0418 and all other lots are "
          "not affected.\nRequired actions for every order of this product from an affected lot:\n"
          "- pending orders: cancel the order (reason: recall RN-2026-14);\n"
          "- shipped or delivered orders: refund the remaining refundable amount of the order and email the "
          "customer with 'RN-2026-14' in the subject, asking them to stop using the product;\n"
          "- cancelled orders and orders already fully refunded: no action.\n"
          "Do not take any action on orders from unaffected lots.")
    traps = ["unaffected_lot", "status_dependent_action", "pagination"]
    if d >= 2:
        traps += ["legacy_sku", "already_refunded", "same_customer_twice"]
    if d >= 3:
        traps += ["partial_prior_refund", "alias_only_in_sku_changes"]
    prompt = "Apply recall notice RN-2026-14 (internal document 'recall_RN-2026-14') completely. Call finish when done."
    return prompt, exp, traps


# -- Template 3: damaged-stock reconciliation -------------------------------------

def _damage(g: _Gen, d: int):
    rng = g.rng
    skus = rng.sample([k for k in CATALOG if k not in ("LMP-020", "MUG-001")], 6)
    whs = list(WAREHOUSES)
    inc = iter(rng.sample(range(300, 999), 12))
    reports, exp_adj = [], {}

    def add(sku, wh, qty):
        key = f"{sku}@{wh}"
        exp_adj[key] = exp_adj.get(key, 0) - qty

    staff = ["Karim (WH-Lyon)", "Sophie (WH-Lille)", "Yann (WH-Nantes)"]
    a, b = next(inc), next(inc)
    q1, w1 = rng.randint(2, 6), rng.choice(whs)
    reports.append((f"Damage INC-{a}", f"INC-{a}: {q1} units of {skus[0]} damaged by a forklift at {w1}.", 6))
    add(skus[0], w1, q1)
    q2, w2 = rng.randint(1, 4), rng.choice(whs)
    reports.append((f"Damage INC-{b}", f"INC-{b}: {q2} units of {skus[1]} found broken during inventory at {w2}.", 5))
    q2c = q2 + rng.randint(1, 3)
    reports.append((f"Correction INC-{b}", f"Correction for INC-{b}: after a recount the damage is {q2c} units of "
                                            f"{skus[1]}, not {q2}.", 3))
    add(skus[1], w2, q2c)
    c, w3 = next(inc), rng.choice(whs)
    cartons = rng.randint(2, 3)
    reports.append((f"Damage INC-{c}", f"INC-{c}: {cartons} cartons of MUG-001 dropped from a shelf at the "
                                        f"{WAREHOUSES[w3]}.", 4))
    add("MUG-001", w3, cartons * 6)
    traps = ["correction_replaces", "cartons_to_units", "warehouse_alias", "pagination"]
    if d >= 2:
        e, w4, q4 = next(inc), rng.choice(whs), rng.randint(2, 5)
        reports.append((f"Damage INC-{e}", f"INC-{e}: {q4} units of {skus[2]} damaged by a water leak at {w4}.", 7))
        reports.append((f"Re: Damage INC-{e}", f"Re-sending the INC-{e} report in case the first message was lost: "
                                                 f"{q4} units of {skus[2]} damaged by a water leak at {w4}.", 2))
        add(skus[2], w4, q4)
        f_, w5, q5 = next(inc), rng.choice(whs), rng.randint(2, 6)
        reports.append((f"Damage INC-{f_}", f"INC-{f_}: {q5} units of LMP-010 desk lamps crushed at {w5}.", 5))
        add("LMP-020", w5, q5)
        traps += ["duplicate_report", "legacy_sku"]
    if d >= 3:
        h, w6, q6 = next(inc), rng.choice(whs), rng.randint(3, 8)
        reports.append((f"FYI INC-{h}", f"INC-{h} (already reconciled on 2026-08-29, for your records only): {q6} "
                                         f"units of {skus[3]} damaged at {w6}.", 2))
        i_, w7, q7 = next(inc), rng.choice(whs), rng.randint(2, 4)
        reports.append((f"Damage INC-{i_}", f"INC-{i_}: {q7} cartons of {skus[4]} (4 units per carton) damaged in "
                                             f"transit at {w7}.", 3))
        add(skus[4], w7, q7 * 4)
        traps += ["already_reconciled", "nonstandard_carton"]
    for subject, body, ago in reports:
        g.ticket(rng.choice(staff), subject, body, days_ago=ago)
    prompt = ("Process the open support ticket queue: record every warehouse damage report in inventory, following "
              "the warehouse handbook. Ignore tickets that are not damage reports. Call finish when done.")
    exp = {"refunds": {}, "cancels": [], "adjust": exp_adj, "emails": []}
    return prompt, exp, traps


# -- Difficulty 4: randomly stacked traps, larger volumes ---------------------------

def _q(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _late_delivery_hard(g: _Gen, d: int):
    rng = g.rng
    optional = ["namesake", "promo", "boundary", "prior_refunds", "cap", "ticket_claim"]
    active = set(optional) if d >= 5 else set(rng.sample(optional, 4))
    traps = ["outdated_policy", "pagination", "unread_memo"] + sorted(active)
    name = g.name()
    cid = g.customer(name)
    if "namesake" in active:
        twin = g.customer(name)
        for delay in (6, 12, 4, -1):
            g.order(twin, status="delivered", delay=delay)
    cases = [("late", rng.choice([4, 5, 6, 7, 8, 9])) for _ in range(5)]
    cases += [("late", rng.choice([11, 12, 13, 15])) for _ in range(3)]
    cases += [("ontime", rng.choice([-2, -1, 0, 1, 2])) for _ in range(4)]
    cases += [("window", rng.choice([5, 6, 7])), ("window", rng.choice([12, 13, 14])), ("window", 3)]
    cases += [("pending", None), ("shipped", None), ("cancelled", None)]
    if "promo" in active:
        cases += [("promo", 9)]
    if "boundary" in active:
        cases += [("late", 3), ("late", 10)]
    if "prior_refunds" in active:
        cases += [("prior_small", 8), ("prior_big", 5)]
    if "cap" in active:
        cases += [("prior_late_comp", 6)]
    rng.shuffle(cases)
    window = (date(2026, 6, 20), date(2026, 7, 10))
    outside = list(range(30, 50)) + list(range(76, 170))
    eligible, prior_late = [], Decimal("0")
    ontime_ids, records = [], []
    for kind, delay in cases:
        status = {"pending": "pending", "cancelled": "cancelled", "shipped": "shipped"}.get(kind, "delivered")
        ago = rng.randint(1, 3) if status == "pending" else rng.randint(54, 72) if kind == "window" \
            else rng.choice(outside)
        oid = g.order(cid, status=status, delay=delay if status == "delivered" else 0, days_ago=ago,
                      promo="SUMMER15" if kind == "promo" else None, qty=rng.choice([1, 2, 3]),
                      sku=rng.choice([k for k in CATALOG if k not in ("CBL-030", "MUG-001")]))
        o = g.s["orders"][oid]
        prior = Decimal("0")
        if kind == "prior_small":
            prior = _q(o["total"] * 0.05)
            g.refund(oid, float(prior), reason="goodwill gesture")
        elif kind == "prior_big":
            prior = _q(o["total"] * 0.25)
            g.refund(oid, float(prior), reason="damaged item")
        elif kind == "prior_late_comp":
            prior = _q(Decimal(str(o["total"])) * Decimal("0.15"))
            g.refund(oid, float(prior), days_ago=rng.randint(80, 120), reason="late delivery")
            prior_late += prior
        if kind == "ontime":
            ontime_ids.append(oid)
        records.append((oid, kind, delay, prior, status))
    slips, slip_atts, rule8 = {}, [], ""
    if d >= 6:  # signed delivery slips: order id only in the QR code, true date only handwritten
        traps.append("delivery_slips_qr_handwriting")

        def pick(pred):
            return rng.choice([r for r in records if r[4] == "delivered" and pred(r)])

        picks = [(pick(lambda r: r[1] == "ontime" and r[0] != ontime_ids[0]), rng.randint(5, 8)),
                 (pick(lambda r: r[1] == "late" and r[2] > 10), rng.randint(6, 9)),
                 (pick(lambda r: r[1] == "late" and 3 < r[2] <= 9), rng.randint(0, 2))]
        for (oid, *_), new_delay in picks:
            o = g.s["orders"][oid]
            true_date = date.fromisoformat(o["promised_date"]) + timedelta(days=new_delay)
            slips[oid] = true_date.isoformat()
            carrier = rng.choice(["DPD", "Colissimo"])
            qr = f"SHIPMENT|ORDER={oid}|CARRIER={carrier}|SCAN={o['delivered_date']}"
            slip_atts.append(g.attach("delivery_slip", {"carrier": carrier, "recipient": name, "qr": qr,
                                                        "hand_date": true_date.strftime("%d/%m/%Y")}, [qr]))
        rng.shuffle(slip_atts)
        rule8 = ("\n8. When the customer provides a signed delivery slip for an order, the delivery date of that "
                 "order is the date handwritten on the slip (carrier tracking dates can be wrong).")
    for oid, kind, delay, prior, status in records:
        if status != "delivered" or kind == "promo":
            continue
        o = g.s["orders"][oid]
        od = date.fromisoformat(o["order_date"])
        delivered = slips.get(oid, o["delivered_date"])
        delay = (date.fromisoformat(delivered) - date.fromisoformat(o["promised_date"])).days
        eff = delay - 4 if window[0] <= od <= window[1] else delay
        pct = Decimal("0.30") if eff > 10 else Decimal("0.15") if eff > 3 else Decimal("0")
        comp = _q(Decimal(str(o["total"])) * pct) - prior
        if comp > 0:
            eligible.append((delivered, oid, comp))
    eligible.sort()
    expected = {}
    rule7 = ""
    if "cap" in active:
        cap = int(prior_late + Decimal("0.6") * sum(c for _, _, c in eligible))
        remaining = Decimal(cap) - prior_late
        for _, oid, comp in eligible:
            amt = min(comp, remaining)
            if amt > 0:
                expected[oid] = float(amt)
                remaining -= amt
        rule7 = (f"\n7. The total late-delivery compensation granted to a customer in calendar year 2026 may not exceed "
                 f"EUR {cap}; refunds already issued in 2026 with reason 'late delivery' count toward this limit. "
                 "When the limit would be exceeded, compensate eligible orders in order of delivery date (oldest "
                 "first; same date: lower order number first) and grant the last one only the amount remaining under "
                 "the limit.")
    else:
        expected = {oid: float(comp) for _, oid, comp in eligible}
    total_comp = sum(Decimal(str(v)) for v in expected.values())
    g.doc("late_delivery_policy", "Late delivery compensation policy", POLICY_V3)
    g.doc("late_delivery_policy_update_2026", "Late delivery policy - 2026 update", POLICY_V4 + rule7 + rule8)
    g.doc("ops_memo_2026-06-12", "Operations memo 2026-06-12",
          "Following the carrier strike, the promised delivery date of every order placed between 2026-06-20 and "
          "2026-07-10 (inclusive) is extended by 4 days when assessing late-delivery compensation.")
    c = g.s["customers"][cid]
    first = name.split(" ")[0]
    claim = (f" For instance order {ontime_ids[0]} arrived almost three weeks late." if "ticket_claim" in active
             else "")
    slip_text = (" I attach photos of the signed delivery slips of three orders: the carrier tracking dates are "
                 "wrong for those." if slip_atts else "")
    tid = g.ticket(f"{name} <{c['email']}>", "Late deliveries - compensation",
                   f"Hello,\nSeveral of my orders this year arrived later than the promised date.{claim}{slip_text} "
                   f"Could you please apply the late-delivery compensation I am entitled to under your policy?\n"
                   f"Thank you,\n{first}", days_ago=1, attachments=slip_atts or None)
    prompt = (f"Handle support ticket {tid} end to end, strictly following company policy and any operational memo "
              "that applies (issue every refund that is due and inform the customer). Call finish when done.")
    variants = [f"{total_comp:.2f}"]
    if variants[0].endswith("0"):
        variants.append(variants[0][:-1])
    variants += [x.replace(".", ",") for x in variants]
    exp = {"refunds": expected, "cancels": [], "adjust": {}, "emails": [{"to": c["email"], "must_contain_any": variants}]}
    return prompt, exp, traps


def _recall_hard(g: _Gen, d: int):
    rng = g.rng
    optional = ["legacy_sku", "bundle", "lot_range", "amendment_memo", "partial_refund", "already_refunded"]
    active = set(optional) if d >= 5 else set(rng.sample(optional, 4))
    traps = ["pagination", "status_dependent_action", "unaffected_lot"] + sorted(active)
    if "lot_range" in active:
        affected = {"K-0415", "K-0416", "K-0417", "K-0419", "K-0420", "K-0421"}
        lots = ["K-0414", "K-0415", "K-0417", "K-0418", "K-0420", "K-0421", "K-0422"]
        lot_text = "Affected production lots: K-0415 to K-0421 inclusive, except lot K-0418. All other lots are not affected."
    else:
        affected = {"K-0417", "K-0419"}
        lots = ["K-0417", "K-0418", "K-0419", "K-0322"]
        lot_text = "Affected production lots: K-0417 and K-0419. Lot K-0418 and all other lots are not affected."
    if "amendment_memo" in active:
        affected = affected | {"K-0423"}
        lots.append("K-0423")
    codes = [("KET-200B", 16 if d >= 5 else 12, (10, 170))]
    if "legacy_sku" in active:
        codes.append(("KET-200", 5, (190, 240)))
    if "bundle" in active:
        codes.append(("BND-K01", 5, (10, 170)))
    cids = list(g.s["customers"])
    twin_cid = rng.choice(cids)
    plan = []
    for sku, n, (lo, hi) in codes:
        aff = sorted(l for l in lots if l in affected)
        unaff = sorted(l for l in lots if l not in affected)
        rows = [(sku, rng.choice(aff), "pending"), (sku, rng.choice(aff), "delivered"),
                (sku, rng.choice(unaff), rng.choice(["pending", "delivered"]))]
        if "K-0423" in lots:
            rows.append((sku, "K-0423", rng.choice(["delivered", "shipped"])))
        while len(rows) < n:
            rows.append((sku, rng.choice(lots), rng.choices(["pending", "shipped", "delivered", "cancelled"],
                                                            [3, 2, 5, 1])[0]))
        if sku == "KET-200":  # legacy code: only historical orders, none still pending
            rows = [(s, l, "delivered" if st == "pending" else st) for s, l, st in rows]
        plan += [(s, l, st, lo, hi) for s, l, st in rows]
    aff_delivered = [i for i, p in enumerate(plan) if p[1] in affected and p[2] == "delivered"]
    extras = {}
    picks = rng.sample(aff_delivered, min(len(aff_delivered), 4))
    if "already_refunded" in active:
        extras[picks.pop()] = "full_refund"
    if "partial_refund" in active:
        extras[picks.pop()] = "partial_refund"
    twins = picks[:2]
    exp = {"refunds": {}, "cancels": [], "adjust": {}, "emails": []}
    emails = set()
    hidden, relabelled = set(), set()
    if d >= 6:  # lot only readable from the packing-label QR code; printed lot is stale on relabelled boxes
        traps += ["lot_in_qr_only", "relabelled_box"]
        hidden = set(rng.sample(range(len(plan)), len(plan) // 2))
        relabelled = set(rng.sample(sorted(hidden), 3))
    for i, (sku, lot, status, lo, hi) in enumerate(plan):
        cid = twin_cid if i in twins else rng.choice(cids)
        ago = rng.randint(1, 4) if status == "pending" and sku != "KET-200" else rng.randint(lo, hi)
        oid = g.order(cid, sku=sku, status=status, lot=lot, delay=rng.choice([-1, 0, 1]), days_ago=ago)
        o = g.s["orders"][oid]
        if i in hidden:
            printed = lot
            if i in relabelled:
                printed = rng.choice(sorted(l for l in lots if (l in affected) != (lot in affected)))
            qr = f"MAISONCO|SKU={sku}|LOT={lot}|ORDER={oid}"
            spec = {"product": CATALOG[LEGACY_SKUS.get(sku, sku)][0], "printed_lot": printed, "qr": qr,
                    "packed": o["order_date"], "stamp": "RELABELLED" if i in relabelled else None}
            o["lot"] = None
            o["attachments"] = [g.attach("packing_label", spec, [qr])]
        refunded = 0.0
        if extras.get(i) == "full_refund":
            g.refund(oid, o["total"], reason="customer return")
            refunded = o["total"]
        elif extras.get(i) == "partial_refund":
            refunded = _money(o["total"] * 0.3)
            g.refund(oid, refunded, reason="goodwill gesture")
        if lot not in affected:
            continue
        if status == "pending":
            exp["cancels"].append(oid)
        elif status in ("shipped", "delivered") and _money(o["total"] - refunded) > 0:
            exp["refunds"][oid] = _money(o["total"] - refunded)
            emails.add(g.s["customers"][cid]["email"])
    exp["emails"] = [{"to": e, "must_contain_any": ["RN-2026-14"]} for e in sorted(emails)]
    bundle_line = ("\nThis notice also covers the kettle sold within the breakfast bundle BND-K01 (same production "
                   "lots); apply the same actions to those orders." if "bundle" in active else "")
    g.doc("recall_RN-2026-14", "Recall notice RN-2026-14",
          f"Recall notice RN-2026-14 - Electric kettle 1.7L (SKU KET-200B).\nIssued 2026-08-28. A thermostat defect "
          f"may cause overheating.\n{lot_text}{bundle_line}\nRequired actions for every order of this product from an "
          "affected lot:\n- pending orders: cancel the order (reason: recall RN-2026-14);\n"
          "- shipped or delivered orders: refund the remaining refundable amount of the order and email the "
          "customer with 'RN-2026-14' in the subject, asking them to stop using the product;\n"
          "- cancelled orders and orders already fully refunded: no action.\n"
          "Do not take any action on orders from unaffected lots."
          + ("\nIf an order has no lot recorded, its lot is given by the QR code of the packing label photo attached "
             "to the order. Printed lot numbers on relabelled boxes may be outdated: the QR code is authoritative."
             if d >= 6 else ""))
    if "amendment_memo" in active:
        g.doc("quality_memo_QN-88", "Quality memo QN-88",
              "Further to recall notice RN-2026-14: lab tests show that production lot K-0423 has the same thermostat "
              "defect. Lot K-0423 is therefore also affected; apply the RN-2026-14 actions to it.")
    prompt = ("Apply recall notice RN-2026-14 (internal document 'recall_RN-2026-14') completely, taking into account "
              "any related internal document. Call finish when done.")
    return prompt, exp, traps


def _damage_hard(g: _Gen, d: int):
    rng = g.rng
    optional = ["duplicate_report", "legacy_sku", "already_reconciled", "nonstandard_carton", "warehouse_alias",
                "transit_destination", "repairable_followup", "correction_chain", "words_quantity"]
    active = set(optional) | {"partial_repair", "retracted", "correction_by_number"} if d >= 5         else set(rng.sample(optional, 5))
    traps = ["pagination", "correction_replaces", "cartons_to_units"] + sorted(active)
    pool = [k for k in CATALOG if k not in ("LMP-020", "MUG-001", "BND-K01")]
    whs = list(WAREHOUSES)
    inc = iter(rng.sample(range(300, 999), 30))
    staff = ["Karim (WH-Lyon)", "Sophie (WH-Lille)", "Yann (WH-Nantes)"]
    reports, exp_adj = [], {}

    def add(sku, wh, qty):
        key = f"{sku}@{wh}"
        exp_adj[key] = exp_adj.get(key, 0) - qty
        if exp_adj[key] == 0:
            del exp_adj[key]

    def where(wh):
        return f"the {WAREHOUSES[wh]}" if "warehouse_alias" in active and rng.random() < 0.5 else wh

    def report(subject, body, ago, attachments=None):
        reports.append((subject, body, ago, attachments))

    for _ in range(7 if d >= 5 else 4):
        n_, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(1, 7)
        report(f"Damage INC-{n_}", f"INC-{n_}: {q} units of {sku} damaged at {where(wh)} "
                                  f"({rng.choice(['forklift', 'fallen shelf', 'water leak', 'crushed pallet'])}).",
               rng.randint(4, 12))
        add(sku, wh, q)
    b, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(1, 4)
    q2 = q + rng.randint(1, 3)
    report(f"Damage INC-{b}", f"INC-{b}: {q} units of {sku} found broken during inventory at {where(wh)}.", 9)
    report(f"Correction INC-{b}", f"Correction for INC-{b}: after a recount it is {q2} units, not {q}."
           if "correction_by_number" in active else
           f"Correction for INC-{b}: after a recount the damage is {q2} units of {sku}, not {q}.", 6)
    if "correction_chain" in active:
        q3 = q2 - 1
        report(f"Correction 2 INC-{b}", f"Second correction for INC-{b}: final count is {q3} units of {sku} "
                                        "(one unit was counted twice).", 3)
        q2 = q3
    add(sku, wh, q2)
    c, wh = next(inc), rng.choice(whs)
    cartons = rng.randint(2, 3)
    report(f"Damage INC-{c}", f"INC-{c}: {cartons} cartons of MUG-001 dropped from a shelf at the "
                              f"{WAREHOUSES[wh]}.", 8)
    add("MUG-001", wh, cartons * 6)
    if "duplicate_report" in active:
        e, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(2, 5)
        report(f"Damage INC-{e}", f"INC-{e}: {q} units of {sku} damaged by a water leak at {wh}.", 10)
        report(f"Fwd: Damage INC-{e}", f"Forwarding again the INC-{e} report, not sure it reached you: {q} units of "
                                       f"{sku} damaged by a water leak at {wh}.", 2)
        add(sku, wh, q)
    if "legacy_sku" in active:
        f_, wh, q = next(inc), rng.choice(whs), rng.randint(2, 6)
        report(f"Damage INC-{f_}", f"INC-{f_}: {q} units of LMP-010 desk lamps crushed at {where(wh)}.", 5)
        add("LMP-020", wh, q)
    if "already_reconciled" in active:
        h, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(3, 8)
        report(f"FYI INC-{h}", f"INC-{h} (already reconciled on 2026-08-29, for your records only): {q} units of "
                               f"{sku} damaged at {wh}.", 2)
    if "nonstandard_carton" in active:
        i_, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(2, 4)
        report(f"Damage INC-{i_}", f"INC-{i_}: {q} cartons of {sku} (4 units per carton) damaged at {where(wh)}.", 7)
        add(sku, wh, q * 4)
    if "transit_destination" in active:
        j, sku = next(inc), rng.choice(pool)
        src, dst = rng.sample(whs, 2)
        q = rng.randint(2, 5)
        report(f"Damage INC-{j}", f"INC-{j}: {q} units of {sku} damaged in transit from {src} to {dst} "
                                  f"(reported by {src} staff).", 4)
        add(sku, dst, q)
    if "repairable_followup" in active:
        k_, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(2, 5)
        report(f"Damage INC-{k_}", f"INC-{k_}: {q} units of {sku} damaged at {wh} (scratched casing).", 11)
        report(f"Update INC-{k_}", f"Update on INC-{k_}: the {q} units of {sku} were repaired by the technician and "
                                   "put back into stock.", 3)
    if "partial_repair" in active:
        p_, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(4, 7)
        fixed = rng.randint(1, q - 2)
        report(f"Damage INC-{p_}", f"INC-{p_}: {q} units of {sku} damaged at {wh} (cracked screens).", 10)
        report(f"Update INC-{p_}", f"Update on INC-{p_}: {fixed} of the {q} damaged units were repaired and put back "
                                   "into stock; the others are a write-off.", 2)
        add(sku, wh, q - fixed)
    if "retracted" in active:
        r_, sku, wh, q = next(inc), rng.choice(pool), rng.choice(whs), rng.randint(2, 6)
        report(f"Damage INC-{r_}", f"INC-{r_}: {q} units of {sku} damaged at {wh}.", 8)
        report(f"Disregard INC-{r_}", f"Please disregard INC-{r_}: it was sent by mistake during a training "
                                      "session, nothing was damaged.", 5)
    if "words_quantity" in active:
        m_, sku, wh = next(inc), rng.choice(pool), rng.choice(whs)
        report(f"Damage INC-{m_}", f"INC-{m_}: a dozen units of {sku} damaged at {where(wh)} after a pallet fell.", 6)
        add(sku, wh, 12)
    prompt_extra = ""
    if d >= 6:  # photographed paper forms: product only in the label QR code, handwritten manager corrections
        traps += ["damage_forms_qr", "handwritten_correction"]
        g.doc("damage_form_guide", "Guide to damage forms",
              "Warehouse staff may declare damage on a paper form, photographed and attached to a ticket.\n"
              "- The product is identified only by the product label QR code printed on the form: scan it.\n"
              "- Quantities on forms follow the warehouse handbook (cartons, current SKU codes, warehouse names).\n"
              "- A handwritten correction of the quantity, initialled by the warehouse manager, replaces the printed "
              "quantity.")
        prompt_extra = " and the guide to damage forms"
        for variant in ("plain", "cartons", "correction", "legacy"):
            n_, wh = next(inc), rng.choice(whs)
            sku = "LMP-010" if variant == "legacy" else rng.choice(pool)
            current = LEGACY_SKUS.get(sku, sku)
            q = rng.randint(2, 7)
            qty_text, correction = f"{q} units", None
            if variant == "cartons":
                q = rng.randint(2, 3)
                qty_text, q = f"{q} cartons", q * 6
            elif variant == "correction":
                q = q + rng.choice([-1, 2, 3])
                correction = {"hand": str(q), "initials": "M.D. manager"}
            spec = {"title": "DAMAGE FORM", "incident": f"INC-{n_}",
                    "warehouse": WAREHOUSES[wh] if rng.random() < 0.5 else wh,
                    "date": (TODAY - timedelta(days=rng.randint(3, 9))).isoformat(), "qty": qty_text,
                    "cause": rng.choice(["forklift", "fallen shelf", "water leak", "crushed pallet"]),
                    "qr": f"MAISONCO|SKU={sku}|DESC={CATALOG[current][0]}", "correction": correction}
            aid = g.attach("damage_form", spec, [spec["qr"]])
            report(f"Damage form INC-{n_}", f"Damage form attached (INC-{n_}).", rng.randint(2, 9), [aid])
            add(current, wh, q)
    for _ in range(4):
        c_ = g.s["customers"][rng.choice(list(g.s["customers"]))]
        subject, body = rng.choice([("Damaged parcel", "My order arrived with a dented box, the item works though."),
                                    ("Broken item", "The mug set I received has one broken mug, what can I do?"),
                                    ("Late refund", "I returned an item two weeks ago and have no refund yet.")])
        g.ticket(f"{c_['name']} <{c_['email']}>", subject, body, rng.randint(1, 12))
    for subject, body, ago, atts in reports:
        g.ticket(rng.choice(staff), subject, body, days_ago=ago, attachments=atts)
    prompt = ("Process the open support ticket queue: record every warehouse damage report in inventory, following "
              f"the warehouse handbook{prompt_extra}. Ignore tickets that are not warehouse damage reports. "
              "Call finish when done.")
    exp = {"refunds": {}, "cancels": [], "adjust": exp_adj, "emails": []}
    return prompt, exp, traps


def _dispatch(easy, hard):
    return lambda g, d: hard(g, d) if d >= 4 else easy(g, d)


TEMPLATES = {"late_delivery": _dispatch(_late_delivery, _late_delivery_hard),
             "recall": _dispatch(_recall, _recall_hard),
             "damage": _dispatch(_damage, _damage_hard)}


def generate(template: str, seed: int, difficulty: int) -> dict:
    g = _Gen(random.Random(f"{template}:{seed}:{difficulty}"))
    g.base_world()
    prompt, expected, traps = TEMPLATES[template](g, difficulty)
    state = g.s
    return {"task_id": f"{template}-s{seed}-d{difficulty}", "template": template, "seed": seed,
            "difficulty": difficulty, "prompt": prompt, "state": state, "expected": expected, "traps": traps,
            "state_hash": state_hash(state)}


def state_hash(state: dict) -> str:
    return hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


def new_world(task: dict) -> World:
    return World(copy.deepcopy(task["state"]))


# ---------------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------------

def check(task: dict, world: World) -> tuple[bool, list[str]]:
    """Compare the final world state with the expected effects. Returns (passed, violations)."""
    exp, s, v = task["expected"], world.s, []
    got_refunds = {}
    for r in s["refunds"]:
        if r["by"] == "agent":
            got_refunds[r["order_id"]] = _money(got_refunds.get(r["order_id"], 0) + r["amount"])
    for oid, amt in exp["refunds"].items():
        if oid not in got_refunds:
            v.append(f"missing refund on {oid}")
        elif abs(got_refunds[oid] - amt) > 0.011:
            v.append(f"wrong refund amount on {oid}")
    v += [f"unexpected refund on {oid}" for oid in got_refunds if oid not in exp["refunds"]]
    got_cancels = {e["order_id"] for e in s["agent_log"] if e["op"] == "cancel"}
    v += [f"missing cancellation of {oid}" for oid in exp["cancels"] if oid not in got_cancels]
    v += [f"unexpected cancellation of {oid}" for oid in got_cancels if oid not in exp["cancels"]]
    got_adj = {}
    for e in s["agent_log"]:
        if e["op"] == "adjust":
            key = f"{e['sku']}@{e['warehouse']}"
            got_adj[key] = got_adj.get(key, 0) + e["delta"]
    for key in set(exp["adjust"]) | set(got_adj):
        if exp["adjust"].get(key, 0) != got_adj.get(key, 0):
            v.append(f"wrong net inventory adjustment for {key}")
    for rule in exp["emails"]:
        ok = any(m["to"].lower() == rule["to"].lower()
                 and any(t in m["subject"] + " " + m["body"] for t in rule["must_contain_any"])
                 for m in s["outbox"])
        if not ok:
            v.append(f"missing or incomplete email to {rule['to']}")
    return not v, v


def oracle_actions(task: dict) -> list[tuple[str, dict]]:
    """A correct action sequence derived from the expected effects (used to validate the generator)."""
    exp, acts = task["expected"], []
    acts += [("cancel_order", {"order_id": oid, "reason": "recall"}) for oid in exp["cancels"]]
    acts += [("issue_refund", {"order_id": oid, "amount": amt, "reason": "policy"}) for oid, amt in exp["refunds"].items()]
    for key, delta in exp["adjust"].items():
        sku, wh = key.split("@")
        acts.append(("adjust_inventory", {"sku": sku, "warehouse": wh, "delta": delta, "reason": "damage"}))
    acts += [("send_email", {"to": e["to"], "subject": e["must_contain_any"][0], "body": "Total: " + e["must_contain_any"][0]})
             for e in exp["emails"]]
    acts.append(("finish", {"summary": "done"}))
    return acts
