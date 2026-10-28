import json
import os
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

AGENT_CARD_URL = os.getenv(
    "ORDERS_AGENT_CARD_URL",
    "http://127.0.0.1:8002/.well-known/agent-card.json"
)

REQUEST_TIMEOUT = 10.0

class RemoteAgentError(RuntimeError):
    "Raised when discovery or a remote agent call fails"

def discover_agent() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    "Fethes the agent card, hten discovers tools through JSON-RPC"
    try:
        with httpx.Client(timeout = REQUEST_TIMEOUT) as client:
            card_response = client.get(AGENT_CARD_URL)
            card_response.raise_for_status()
            card = card_response.json()

            rpc_url = card.get("url")
            if not rpc_url:
                raise RemoteAgentError("Agent card does not contain a URL")

            tools_response = client.post(
                rpc_url,
                json= {
                    "jsonrpc":"2.0",
                    "id":"discovery-1",
                    "method":"agent.list_tools",
                    "params":{}
                }
            )

            tools_response.raise_for_status()
            payload = tools_response.json()

    except (httpx.HTTPError, ValueError) as e:
        raise RemoteAgentError(f"Orders agent discovery failed:{e}") from e

    except httpx.TimeoutException:
        raise RemoteAgentError("the agent card url didnt responsd in time.")

    if "error" in payload:
        raise RemoteAgentError(f"Tool discovery returned an RPC error: {payload["error"]}")

    result = payload.get("result", {})
    tools = result.get("tools", []) if isinstance(result, dict) else []

    if not tools:
        raise RemoteAgentError(
            "No tools were returned"
        )

    return card, tools

def _python_type(json_type:str) -> type:
    "Map basic JSON schema types to python types"
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }.get(json_type)

def _build_args_schema(
    tool_name: str,
    tool_definition: dict[str, Any]
) -> type[BaseModel]:
    "Build a pydantic input model from the remote tool's json schema"
    schema=(
        tool_definition.get("inputSchema")
        or tool_definition.get("input_schema")
        or tool_definition.get("parameters")
        or {"type":"object", "properties":{}}
    )

    properties = schema.get("properties",{})
    required_fields = set(schema.get("required",[]))
    fields: dict[str, Any]= {}

    for field_name, field_schema in properties.items():
        python_type = _python_type(field_schema.get("type","string"))
        description = field_schema.get("description", "")
        allowed_enum = field_schema.get("enum", None)
        print(f"==========allowed enum is==========:{allowed_enum}===type:{type(allowed_enum)}")
        if allowed_enum:
            python_type = Literal[allowed_enum]

        if field_name in required_fields:
            fields[field_name]=(
                python_type,
                Field(..., description = description)
            )
        else:
            default = field_schema.get("default", None)
            fields[field_name]=(
                python_type | None,
                Field(default = default, description = description)
            )
    
    return create_model(f"{tool_name.title().replace("-","")}Args", **fields)

def build_remote_tools(
    authorization:str
) -> tuple[dict[str,Any],list[StructuredTool]]:
    """Discover tools and create langchain tools that call the remote agents
    authorization is caotured unchanged and forwarded to the other agents
    """
    card, definitions = discover_agent()
    rpc_url = card["url"]
    langchain_tools: list[StructuredTool]=[]

    for definition in definitions:
        tool_name = definition["name"]
        description = definition.get("description", f"Remote tool: {tool_name}")
        args_schema = _build_args_schema(tool_name, definition)

        def make_remote_call(
            name: str,
            endpoint: str
        ):
            def invoke(**arguments: Any) -> str:
                
                clean_arguments = {
                    key: value
                    for key, value in arguments.items()
                    if value is not None
                }

                try:
                    with httpx.Client(timeout = REQUEST_TIMEOUT) as client:
                        response = client.post(
                            endpoint,
                            headers= {"Authorization":authorization},
                            json={
                                "jsonrpc": "2.0",
                                "id": f"tool-{name}",
                                "method":"agent.call_tool",
                                "params":{
                                    "name": name,
                                    "arguments": clean_arguments
                                }
                            }

                        )
                        response.raise_for_status()

                        payload = response.json()
                
                except httpx.HTTPError as e:
                    return json.dumps({
                        "ok": False,
                        "error": f"Remote tool request failed: {e}"
                    })
                if "error" in payload:
                    return json.dumps({
                        "ok":False,
                        "error":payload["error"]
                    })

                # The remote tool output is unstructured data. the model should
                # trat as evidence, never as instructions
                return json.dumps(
                    {"ok":True, "result": payload.get("result")},
                    ensure_ascii = False,
                    default = str
                )

            return invoke
        
        langchain_tools.append(
            StructuredTool.from_function(
                func = make_remote_call(tool_name, rpc_url),
                name = tool_name,
                description = description,
                args_schema = args_schema
            )
        )
    print(f"=========langchain tools are:============={langchain_tools}")
    return card, langchain_tools