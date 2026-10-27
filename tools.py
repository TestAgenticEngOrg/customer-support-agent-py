import uuid
from datetime import datetime, timezone

# Mock data. Replace with calls to real order and ticketing systems.
ORDERS = {
    "ORD-1001": {"status": "Shipped", "item": "Wireless Headphones", "estimatedDelivery": "2026-10-29"},
    "ORD-1002": {"status": "Processing", "item": "Mechanical Keyboard", "estimatedDelivery": "2026-11-03"},
    "ORD-1003": {"status": "Delivered", "item": "USB-C Hub", "estimatedDelivery": "2026-10-20"},
}

FAQS = {
    "returns": "Items can be returned within 30 days of delivery in original condition. Refunds take 5-7 business days.",
    "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "warranty": "All electronics include a 1-year limited warranty covering manufacturing defects.",
    "payment": "We accept credit cards, debit cards and PayPal.",
}

tickets: list[dict] = []


def get_order_status(order_id: str) -> dict | str:
    order_id = order_id.strip().upper()
    order = ORDERS.get(order_id)
    return {"orderId": order_id, **order} if order else f"No order found with ID {order_id}."


def search_faq(topic: str) -> str:
    answer = FAQS.get(topic.strip().lower())
    return answer or f"No FAQ found. Available topics: {', '.join(FAQS)}."


def create_support_ticket(summary: str, customer_email: str) -> dict:
    ticket = {
        "ticketId": "TCK-" + uuid.uuid4().hex[:8].upper(),
        "summary": summary,
        "customerEmail": customer_email,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    tickets.append(ticket)
    return ticket


FUNCTIONS = {
    "get_order_status": get_order_status,
    "search_faq": search_faq,
    "create_support_ticket": create_support_ticket,
}


def call(name: str, args: dict):
    fn = FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Invalid arguments for {name}: {e}"


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": list(properties)},
    }


SCHEMAS = [
    _schema("get_order_status", "Looks up the current status of a customer order.",
            {"order_id": {"type": "string", "description": "The order ID, for example ORD-1001"}}),
    _schema("search_faq", "Searches the FAQ knowledge base by topic.",
            {"topic": {"type": "string", "description": "One of: returns, shipping, warranty, payment"}}),
    _schema("create_support_ticket", "Creates a support ticket to escalate an issue to a human agent.",
            {"summary": {"type": "string", "description": "Short description of the customer's issue"},
             "customer_email": {"type": "string", "description": "The customer's email address for follow up"}}),
]
