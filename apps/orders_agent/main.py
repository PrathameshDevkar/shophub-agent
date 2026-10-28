import json
import logging 
import os
import re
from typing import Any
from uuid import uuid4
from dotenv import load_dotenv
import httpx
from fastapi import  FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

load_dotenv(dotenv_path = r"C:\Users\Prathamesh\prathamesh\ai_agent\shophub-agent\.env")

logger = logging.getLogger(__name__)

STORE_API_URL = os.getenv(
    "STORE_API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")

STORE_TIMEOUT_SECONDS = float(
    os.getenv("STORE_TIMEOUT_SECONDS", "5")
)

app = FastAPI(
    title = "Shophub Orders Agent",
    version = "0.1.0",
    description = "Remote order support agent using JSON-RPC 2.0"
)

#JSON-RPC 2.0 schemas

class JsonRpcRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    jsonrpc: str
    method: str
    params: dict[str, Any] = Field(default_factory = dict)
    id: str | int | None = None

def rpc_result(request_id: str | int | None,result: Any ) -> dict:
    return {
        "jsonrpc": "2.0",
        "id" : request_id,
        "result": result
    }

def rpc_error(
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any = None
)-> dict:
    error = {"code":code, "message": message}

    if data is not None:
        error["data"] = data

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error
    }

# Tool description
TOOLS = [
    {
        "name": "list_orders",
        "description": (
            "List orders belonging to the authenticated merchant. "
            "Optionally filter by order status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "PLACED",
                        "PAID",
                        "SHIPPED",
                        "DELIVERED",
                        "CANCELLED",
                        "REFUNDED",
                    ],
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "Look up an order by ID within the authenticated merchant's "
            "store. A missing or foreign order is reported as a failure."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "minimum": 1}
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_refund",
        "description": (
            "Create a refund proposal without executing the refund. "
            "The proposal must be explicitly approved before execution."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "minimum": 1}
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_cancel",
        "description": (
            "Create a cancellation proposal without cancelling the order. "
            "The proposal must be explicitly approved before execution."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "minimum": 1}
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
]

# Tool output guardrails
INSTRUCTION_LIKE_PATTERNS = [
    r"\bignore\b.{0,80}\b(instructions|rules|prompt|previous)\b",
    r"\b(assistant|system)\s*:",
    r"\b(reveal|expose)\b.{0,40}\b(system prompt|secrets|credentials)\b",
    r"\brefund\b.{0,60}\b(immediately|without approval)\b",
    r"\boverride\b.{0,40}\b(instructions|rules|guardrails)\b",
]

INSTRUCTION_LIKE_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in INSTRUCTION_LIKE_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL,
)

def sanitize_customer_note(note: str) -> str:
    """Treat suspicious customer notes as untrusted data, not instructions."""
    if INSTRUCTION_LIKE_RE.search(note):
        return "[Instruction-like customer note removed.]"

    return note

def model_safe_order(order: dict[str, Any]) -> dict[str, Any]:
    """
    Return only fields needed for order support.

    Deliberately exclude customer_email and customer_phone so they cannot
    reach the model through these tool results.
    """
    safe_order = {
        "id": order["id"],
        "merchant": order["merchant"],
        "customer_name": order["customer_name"],
        "status": order["status"],
        "total": order["total"],
        "placed_at": order["placed_at"],
        "customer_note": sanitize_customer_note(
            order.get("customer_note", "")
        ),
    }

    return safe_order

# Store API client

def store_request(
    method: str,
    path: str,
    authorization: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None
) -> dict:
    """
    Make a store API request using the caller's original Authorization
    header. No credentials are generated, elevated or cached.
    """
    print("=========inside the store_request response is=============")
    try:
        print("============inside the try===================")
        with httpx.Client(timeout = STORE_TIMEOUT_SECONDS) as client:
            response = client.request(
                method = method,
                url = f"{STORE_API_URL}{path}",
                headers= {"Authorization": authorization},
                params = params,
                json= json_body
            )
        print(f"=========inside the store_request response is=============:{response}")
    
    except httpx.TimeoutException:
        return {
            "ok": False,
            "error":{
                "type": "store_timeout",
                "message":"The store API didnt respons in time"
            }
        }
    
    except httpx.RequestError:
        logger.exception("Store API connection failed")
        return {
            "ok": False,
            "error":{
                "type":"store_unavailable",
                "message": "The store API is currently unavailable."
            }
        }

    try:
        body= response.json()
    except ValueError:
        body= {}

    if not response.is_success:
        return {
            "ok": False,
            "error":{
                "type": "store_api_error",
                "status_code":response.status_code,
                "message": body.get("detail", "Store API request failed.")
            }
        }

    return {"ok":True, "data":body}

def authenticate_merchant(authorization:str) -> dict:
    """
    Resolve identity through store api on every RPC request.
    The store_api remains the source of truth for the caller's merchant
    """
    print("=================inside the authenticate_merchant===============")
    result = store_request("GET", "/auth/me", authorization)
    print(f"=================result is===============:{result}")

    if not result["ok"]:
        error = result["error"]
        status = error.get("status_code", 502)

        if status not in (401, 403):
            status = 502

        raise HTTPException(status_code = status, detail= error["message"])
    
    identity = result["data"]

    if identity.get("merchant") is None:
        raise HTTPException(
            status_code = 403,
            detail = "token is not associated with a merchant"
        )
    
    return identity

def lookup_order(
    order_id: int,
    authorization: str
)-> dict:
    result = store_request(
        "GET",
        "/orders/lookup",
        authorization,
        params = {"id": order_id}
    )

    if not result["ok"]:
        return result

    # this legacy endpoint can return HTTP 200 on failure
    envelope = result["data"]

    if envelope.get("status") is not True:
        return {
            "ok": False,
            "error": {
                "type":"order_not_found",
                "message": "Order not found in your store"
            }
        }
    
    return {
        "ok": True,
        "data": model_safe_order(envelope["order"])
    }

# Tool execution
def call_tool(
    name: str,
    arguments: dict[str,Any],
    authorization: str
) ->dict:
    if name == "list_orders":
        status = arguments.get("status")

        result = store_request(
            "GET",
            "/orders",
            authorization,
            params= {"status":status} if status else None
        )

        if not result["ok"]:
            return result
        
        orders = result["data"].get("orders",[])

        return {
            "ok": True,
            "data": {
                "orders":[model_safe_order(o) for o in orders],
                "count": len(orders)
            }
        }

    if name == "lookup_order":
        return lookup_order(arguments["id"], authorization)

    if name in {"propose_refund", "propose_cancel"}:
        result = lookup_order(arguments["id"], authorization)

        if not result["ok"]:
            return result

        order = result["data"]
        action = (
            "refund" if name == "propose_refund" else "cancel"
        )

        allowed_states = (
            {"PAID"} if action == "refund" else {"PAID", "PLACED"}
        )

        if order["status"] not in allowed_states:
            return {
                "ok": False,
                "error": {
                    "type": "invalid_state",
                    "message":(
                        f"Cannot {action} an order in state {order["status"]}"
                    )
                }
            }
        
        return {
            "ok": True,
            "data":{
                "proposal_id": str(uuid4()),
                "action":action,
                "order_id":order["id"],
                "merchant":order["merchant"],
                "current_status": order["status"],
                "proposed_status":(
                    "REFUNDED" if action == "refund" else "CANCELLED"
                ),
                "amount":order["total"],
                "requires_explicit_approval":True,
                "executed":False
            }
        }
    
    return {
        "ok": False,
        "error":{
            "type": "unknown_tool",
            "message": f"Unknown tool: {name}"
        }
    }
        

# Agent card and json-rpc endpoint

@app.get("/health")
def health():
    return {"status":"ok", "agent":"orders-agent"}

@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "ShopHub Orders Agent",
        "description": (
            "Handles merchant-scoped order lookup, listing and creation of refund/cancellation proposals."
        ),
        "version":"0.1.0",
        "url": os.getenv(
            "ORDERS_AGENT_PUBLIC_URL",
            "http://127.0.0.1:8002/rpc"
        ),
        "protocol":"JSON-RPC 2.0",
        "skills":[
            {
                "id":"list_orders",
                "name": "List orders",
                "description": "List orders for the authenticated merchant"
            },
            {
                "id": "lookup_order",
                "name": "Look up order",
                "description": "Find one order by ID in the merchant's store."
            },
            {
                "id": "propose_refund",
                "name": "Propose refund",
                "description": "Propose a refund without executing it.",
            },
            {
                "id": "propose_cancel",
                "name": "Propose cancellation",
                "description": "Propose cancellation without executing it.",
            },
        ],
        "capabilities": {
            "streaming": False,
        }
    }

@app.post("/rpc")
def rpc(
    request: JsonRpcRequest,
    authorization: str | None = Header(default=None)
):
    request_id = request.id
    print(f"==============request_id are======================: {request_id}")

    if request.jsonrpc != "2.0":
        return rpc_error(request_id, -32600 ,"Invalid Request")

    if request_id is None:
        return rpc_error(
            None,
            -32600,
            "Request id is required"
        )
    
    try:
        print(f"==============request method are======================: {request.method}")
        
        if request.method == "agent.list_tools":
            if request.params:
                return rpc_error(
                    request_id, -32602, "Method does not accept parameters"
                )
            
            return rpc_result(request_id, {"tools":TOOLS})

        if request.method != "agent.call_tool":
            return rpc_error(request_id, -32601, "Method not found")
        print(f"==============request method are======================: {request.method}")
        if not authorization:
            return rpc_error(
                request_id,
                -32001,
                "Missing bearer token"
            )
        
        # Resolve the caller using the original credential. The identity 
        # returned by this chcek is not accetpted from the LLM or params.
        identity = authenticate_merchant(authorization)
        print(f"==============identity are======================: {identity}")
        name = request.params.get("name")
        arguments = request.params.get("arguments", {})
        print(f"==============arguments are======================: {arguments}")

        if not isinstance(name, str) or not isinstance(arguments, dict):
            return rpc_error(
                request_id, -32602, "Invalid tool name or arguments"
            )

        known_tool = next(
            (tool for tool in TOOLS if tool["name"] == name),
            None
        )

        if known_tool is None:
            return rpc_error(request_id, -32602, "Unknown tool")

        # Reject undeclared arguments instead of silently accepting them.
        properties = known_tool["inputSchema"]["properties"]
        required = known_tool["inputSchema"].get("required", [])

        if set(arguments) - set(properties):
            return rpc_error(
                request_id, -32602, "Unexpected tool arguments"
            )
        
        if any(key not in arguments for key in required):
            return rpc_error(
                request_id, -32602, "Missing required tool arguments"
            )
        
        if name == "list_orders":
            if arguments.get("status") is not None and arguments["status"] not in {
                "PLACED", "PAID", "SHIPPED", "DELIVERED",
                "CANCELLED", "REFUNDED",
            }:
                return rpc_error(request_id, -32602, "Invalid order status")

        if name in {"lookup_order", "propose_refund", "propose_cancel"}:
            order_id = arguments.get("id")
            if type(order_id) is not int or order_id <= 0:
                return rpc_error(request_id, -32602, "Invalid order id")

        result = call_tool(name, arguments, authorization)

        # Business/API failures are returned as result data so the orchestrator can reason about
        # them without treating them as JSON-RPC transport failures
        
        return rpc_result(
            request_id,
            {
                "caller":{
                    "user": identity["user"],
                    "merchant": identity["merchant"]
                },
                **result
            }
        )

    except HTTPException as e:
        return rpc_error(
            request_id,
            -32000,
            e.detail,
            {"http_status": e.status_code}
        )

    except Exception:
        logger.exception("Unexpected orders-agent error")
        return rpc_error(
            request_id,
            -32603,
            "Internal error"
        )