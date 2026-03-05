"""CLI for ToolBox-S tool management.

Provides the ``taproot-tools`` console command for pushing, registering,
invoking, listing, inspecting, and deleting tools via the Taproot platform.

Configuration is read from environment variables:
    TAPROOT_BASE_URL    — API gateway URL (required)
    TAPROOT_API_KEY     — API key (required)
    TAPROOT_PROJECT_ID  — project ID (required)
    TAPROOT_DIRECT_MODE — "true" for direct mode (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, NoReturn, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str) -> str:
    """Return an env var or print an error and exit."""
    value = os.environ.get(name, "")
    if not value:
        _die(f"Environment variable {name} is not set.")
    return value


def _die(message: str, code: int = 1) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def _make_client() -> Any:
    """Build a ``TaprootClient`` from environment variables."""
    # Late import so the module can be loaded without triggering heavy deps
    # at parse time (useful for ``--help``).
    from taproot_sdk.client import TaprootClient  # noqa: E402

    base_url = _env("TAPROOT_BASE_URL")
    api_key = _env("TAPROOT_API_KEY")
    project_id = _env("TAPROOT_PROJECT_ID")
    direct_mode = os.environ.get("TAPROOT_DIRECT_MODE", "").lower() == "true"

    return TaprootClient(
        base_url=base_url,
        api_key=api_key,
        project_id=project_id,
        direct_mode=direct_mode,
    )


def _print_json(data: Any) -> None:
    """Pretty-print *data* as indented JSON."""
    print(json.dumps(data, indent=2, default=str))


def _tool_info_to_dict(tool: Any) -> dict[str, Any]:
    """Convert a ``ToolInfo`` frozen dataclass to a plain dict."""
    return {
        "id": tool.id,
        "name": tool.name,
        "project_id": tool.project_id,
        "tool_type": tool.tool_type,
        "status": tool.status,
        "version": tool.version,
        "scope": tool.scope,
        "description": tool.description,
        "tags": list(tool.tags),
        "entry_point": tool.entry_point,
        "endpoint_url": tool.endpoint_url,
        "timeout_ms": tool.timeout_ms,
        "memory_mb": tool.memory_mb,
        "created_at": tool.created_at,
        "updated_at": tool.updated_at,
    }


def _invocation_to_dict(inv: Any) -> dict[str, Any]:
    return {
        "invocation_id": inv.invocation_id,
        "tool_name": inv.tool_name,
        "success": inv.success,
        "result": inv.result,
        "error": inv.error,
        "duration_ms": inv.duration_ms,
    }


def _print_tool_info(tool: Any, *, as_json: bool = False) -> None:
    if as_json:
        _print_json(_tool_info_to_dict(tool))
        return
    print(f"Tool ID:     {tool.id}")
    print(f"Name:        {tool.name}")
    print(f"Type:        {tool.tool_type}")
    print(f"Status:      {tool.status}")
    print(f"Version:     {tool.version}")
    print(f"Scope:       {tool.scope}")
    if tool.description:
        print(f"Description: {tool.description}")
    if tool.tags:
        print(f"Tags:        {', '.join(tool.tags)}")
    if tool.entry_point:
        print(f"Entry point: {tool.entry_point}")
    if tool.endpoint_url:
        print(f"Endpoint:    {tool.endpoint_url}")
    if tool.created_at:
        print(f"Created:     {tool.created_at}")


def _print_tool_table(tool_list: Any, *, as_json: bool = False) -> None:
    if as_json:
        _print_json({
            "project_id": tool_list.project_id,
            "count": tool_list.count,
            "tools": [_tool_info_to_dict(t) for t in tool_list.tools],
        })
        return

    if not tool_list.tools:
        print("No tools found.")
        return

    # Column widths
    header = f"{'ID':<36}  {'NAME':<20}  {'TYPE':<10}  {'STATUS':<12}  {'VER':>3}"
    print(header)
    print("-" * len(header))
    for t in tool_list.tools:
        print(
            f"{t.id:<36}  {t.name:<20}  {t.tool_type:<10}  "
            f"{t.status:<12}  {t.version:>3}"
        )
    print(f"\n{tool_list.count} tool(s) in project {tool_list.project_id}")


def _print_invocation(inv: Any, *, as_json: bool = False) -> None:
    if as_json:
        _print_json(_invocation_to_dict(inv))
        return
    status = "SUCCESS" if inv.success else "FAILED"
    print(f"Invocation:  {inv.invocation_id}")
    print(f"Tool:        {inv.tool_name}")
    print(f"Status:      {status}")
    print(f"Duration:    {inv.duration_ms:.1f} ms")
    if inv.error:
        print(f"Error:       {inv.error}")
    if inv.result is not None:
        print(f"Result:      {json.dumps(inv.result, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_push(args: argparse.Namespace) -> None:
    """Read source file and push a hosted tool."""
    try:
        with open(args.file_path, encoding="utf-8") as fh:
            source_code = fh.read()
    except FileNotFoundError:
        _die(f"File not found: {args.file_path}")
    except OSError as exc:
        _die(f"Cannot read file {args.file_path}: {exc}")

    client = _make_client()
    requirements = [r.strip() for r in args.requirements.split(",") if r.strip()] \
        if args.requirements else None
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else None

    tool = asyncio.run(
        client.push_tool(
            name=args.name,
            source_code=source_code,
            entry_point=args.entry_point,
            description=args.description or "",
            requirements=requirements,
            tags=tags,
            timeout_ms=args.timeout_ms,
            memory_mb=args.memory_mb,
            scope=args.scope,
        )
    )
    _print_tool_info(tool, as_json=args.json)


def _cmd_register(args: argparse.Namespace) -> None:
    """Register an external HTTP tool."""
    client = _make_client()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else None

    tool = asyncio.run(
        client.register_tool(
            name=args.name,
            endpoint_url=args.endpoint_url,
            description=args.description or "",
            http_method=args.http_method,
            auth_type=args.auth_type,
            tags=tags,
        )
    )
    _print_tool_info(tool, as_json=args.json)


def _cmd_invoke(args: argparse.Namespace) -> None:
    """Invoke a tool by name."""
    input_data: dict[str, Any] | None = None

    if args.input and args.input_file:
        _die("Specify --input or --input-file, not both.")

    if args.input:
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError as exc:
            _die(f"Invalid JSON in --input: {exc}")

    if args.input_file:
        try:
            with open(args.input_file, encoding="utf-8") as fh:
                input_data = json.loads(fh.read())
        except FileNotFoundError:
            _die(f"File not found: {args.input_file}")
        except json.JSONDecodeError as exc:
            _die(f"Invalid JSON in {args.input_file}: {exc}")
        except OSError as exc:
            _die(f"Cannot read file {args.input_file}: {exc}")

    client = _make_client()
    result = asyncio.run(client.invoke_tool(args.tool_name, input_data))
    _print_invocation(result, as_json=args.json)


def _cmd_list(args: argparse.Namespace) -> None:
    """List tools."""
    client = _make_client()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else None

    tool_list = asyncio.run(
        client.list_tools(
            tags=tags,
            tool_type=args.type,
            status=args.status,
        )
    )
    _print_tool_table(tool_list, as_json=args.json)


def _cmd_get(args: argparse.Namespace) -> None:
    """Get a tool by ID."""
    client = _make_client()
    tool = asyncio.run(client.get_tool(args.tool_id))
    _print_tool_info(tool, as_json=args.json)


def _cmd_delete(args: argparse.Namespace) -> None:
    """Delete a tool by ID."""
    client = _make_client()
    asyncio.run(client.delete_tool(args.tool_id))
    if args.json:
        _print_json({"deleted": True, "tool_id": args.tool_id})
    else:
        print(f"Deleted tool {args.tool_id}")


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _credential_info_to_dict(cred: Any) -> dict[str, Any]:
    """Convert a ``CredentialInfo`` frozen dataclass to a plain dict."""
    return {
        "id": cred.id,
        "project_id": cred.project_id,
        "tool_definition_id": cred.tool_definition_id,
        "credential_type": cred.credential_type,
        "name": cred.name,
        "status": cred.status,
        "version": cred.version,
        "expires_at": cred.expires_at,
        "created_at": cred.created_at,
        "updated_at": cred.updated_at,
        "created_by": cred.created_by,
    }


def _print_credential_info(cred: Any, *, as_json: bool = False) -> None:
    if as_json:
        _print_json(_credential_info_to_dict(cred))
        return
    print(f"Credential ID: {cred.id}")
    print(f"Name:          {cred.name}")
    print(f"Type:          {cred.credential_type}")
    print(f"Status:        {cred.status}")
    print(f"Version:       {cred.version}")
    print(f"Tool ID:       {cred.tool_definition_id}")
    if cred.expires_at:
        print(f"Expires:       {cred.expires_at}")
    if cred.created_at:
        print(f"Created:       {cred.created_at}")


def _print_credential_table(cred_list: Any, *, as_json: bool = False) -> None:
    if as_json:
        _print_json({
            "project_id": cred_list.project_id,
            "count": cred_list.count,
            "credentials": [_credential_info_to_dict(c) for c in cred_list.credentials],
        })
        return

    if not cred_list.credentials:
        print("No credentials found.")
        return

    header = f"{'ID':<36}  {'NAME':<20}  {'TYPE':<12}  {'STATUS':<10}  {'VER':>3}"
    print(header)
    print("-" * len(header))
    for c in cred_list.credentials:
        print(
            f"{c.id:<36}  {c.name:<20}  {c.credential_type:<12}  "
            f"{c.status:<10}  {c.version:>3}"
        )
    print(f"\n{cred_list.count} credential(s) in project {cred_list.project_id}")


# ---------------------------------------------------------------------------
# MCP server helpers
# ---------------------------------------------------------------------------

def _mcp_server_to_dict(srv: Any) -> dict[str, Any]:
    """Convert an ``MCPServerInfo`` frozen dataclass to a plain dict."""
    return {
        "id": srv.id,
        "project_id": srv.project_id,
        "name": srv.name,
        "description": srv.description,
        "transport_type": srv.transport_type,
        "url": srv.url,
        "capabilities": list(srv.capabilities),
        "tools_discovered": list(srv.tools_discovered),
        "health_status": srv.health_status,
        "last_health_check": srv.last_health_check,
        "tags": list(srv.tags),
        "scope": srv.scope,
        "created_by": srv.created_by,
        "created_at": srv.created_at,
        "updated_at": srv.updated_at,
    }


def _print_mcp_server_info(srv: Any, *, as_json: bool = False) -> None:
    if as_json:
        _print_json(_mcp_server_to_dict(srv))
        return
    print(f"Server ID:   {srv.id}")
    print(f"Name:        {srv.name}")
    print(f"Transport:   {srv.transport_type}")
    print(f"URL:         {srv.url}")
    print(f"Health:      {srv.health_status}")
    print(f"Scope:       {srv.scope}")
    if srv.description:
        print(f"Description: {srv.description}")
    if srv.tags:
        print(f"Tags:        {', '.join(srv.tags)}")
    if srv.capabilities:
        print(f"Capabilities: {', '.join(srv.capabilities)}")
    if srv.tools_discovered:
        print(f"Tools:       {', '.join(srv.tools_discovered)}")
    if srv.created_at:
        print(f"Created:     {srv.created_at}")


def _print_mcp_server_table(srv_list: Any, *, as_json: bool = False) -> None:
    if as_json:
        _print_json({
            "project_id": srv_list.project_id,
            "count": srv_list.count,
            "servers": [_mcp_server_to_dict(s) for s in srv_list.servers],
        })
        return

    if not srv_list.servers:
        print("No MCP servers found.")
        return

    header = (
        f"{'ID':<36}  {'NAME':<20}  {'TRANSPORT':<10}  "
        f"{'HEALTH':<10}  {'SCOPE':<8}"
    )
    print(header)
    print("-" * len(header))
    for s in srv_list.servers:
        print(
            f"{s.id:<36}  {s.name:<20}  {s.transport_type:<10}  "
            f"{s.health_status:<10}  {s.scope:<8}"
        )
    print(f"\n{srv_list.count} server(s) in project {srv_list.project_id}")


# ---------------------------------------------------------------------------
# Credential subcommands
# ---------------------------------------------------------------------------

def _cmd_import_openapi(args: argparse.Namespace) -> None:
    """Import tools from an OpenAPI spec."""
    spec_body = None
    if args.spec_file:
        try:
            with open(args.spec_file, encoding="utf-8") as fh:
                spec_body = json.loads(fh.read())
        except FileNotFoundError:
            _die(f"File not found: {args.spec_file}")
        except json.JSONDecodeError as exc:
            _die(f"Invalid JSON in {args.spec_file}: {exc}")
        except OSError as exc:
            _die(f"Cannot read file {args.spec_file}: {exc}")

    if not args.spec_url and spec_body is None:
        _die("Specify --spec-url or --spec-file.")

    client = _make_client()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else None

    result = asyncio.run(
        client.import_openapi(
            namespace=args.namespace,
            spec_url=args.spec_url,
            spec_body=spec_body,
            base_url=args.base_url,
            tags=tags,
            auth_type=args.auth_type,
            scope=args.scope,
        )
    )

    if args.json:
        _print_json({
            "tools_created": len(result.tools_created),
            "tools_skipped": result.tools_skipped,
            "total_parsed": result.total_parsed,
            "namespace": result.namespace,
            "tool_names": [t.name for t in result.tools_created],
        })
    else:
        print(
            f"Imported {len(result.tools_created)} tools "
            f"from namespace '{result.namespace}'"
        )
        print(f"  Skipped: {result.tools_skipped} (already exist)")
        print(f"  Total parsed: {result.total_parsed}")
        if result.tools_created:
            print("\nCreated tools:")
            for tool in result.tools_created:
                desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
                print(f"  - {tool.name}: {desc}")


def _cmd_set_credential(args: argparse.Namespace) -> None:
    """Store an encrypted credential for a tool."""
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        _die(f"Invalid JSON in --payload: {exc}")

    client = _make_client()
    cred = asyncio.run(
        client.set_tool_credential(
            tool_definition_id=args.tool_id,
            credential_type=args.type,
            name=args.name,
            credential_payload=payload,
            expires_at=args.expires_at if hasattr(args, "expires_at") else None,
        )
    )
    _print_credential_info(cred, as_json=args.json)


def _cmd_list_credentials(args: argparse.Namespace) -> None:
    """List credentials for the current project."""
    client = _make_client()
    cred_list = asyncio.run(
        client.list_credentials(tool_definition_id=args.tool_id)
    )
    _print_credential_table(cred_list, as_json=args.json)


def _cmd_revoke_credential(args: argparse.Namespace) -> None:
    """Revoke a credential."""
    client = _make_client()
    cred = asyncio.run(
        client.revoke_credential(args.credential_id, args.version)
    )
    _print_credential_info(cred, as_json=args.json)


# ---------------------------------------------------------------------------
# MCP server subcommands
# ---------------------------------------------------------------------------

def _cmd_mcp_register(args: argparse.Namespace) -> None:
    """Register an MCP server."""
    client = _make_client()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else None

    srv = asyncio.run(
        client.register_mcp_server(
            name=args.name,
            transport_type=args.transport,
            url=args.url,
            description=args.description or "",
            tags=tags,
        )
    )
    _print_mcp_server_info(srv, as_json=args.json)


def _cmd_mcp_list(args: argparse.Namespace) -> None:
    """List MCP servers."""
    client = _make_client()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else None

    srv_list = asyncio.run(
        client.list_mcp_servers(status=args.status, tags=tags)
    )
    _print_mcp_server_table(srv_list, as_json=args.json)


def _cmd_mcp_get(args: argparse.Namespace) -> None:
    """Get an MCP server by ID."""
    client = _make_client()
    srv = asyncio.run(client.get_mcp_server(args.id))
    _print_mcp_server_info(srv, as_json=args.json)


def _cmd_mcp_delete(args: argparse.Namespace) -> None:
    """Delete an MCP server."""
    client = _make_client()
    asyncio.run(client.delete_mcp_server(args.id))
    if args.json:
        _print_json({"deleted": True, "server_id": args.id})
    else:
        print(f"Deleted MCP server {args.id}")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taproot-tools",
        description="Manage tools on the Taproot ToolBox-S platform.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output machine-readable JSON instead of human-readable text.",
    )

    sub = parser.add_subparsers(dest="command")

    # -- push ---------------------------------------------------------------
    push_p = sub.add_parser("push", help="Push a hosted Python tool.")
    push_p.add_argument("file_path", help="Path to the Python source file.")
    push_p.add_argument("--name", required=True, help="Tool name.")
    push_p.add_argument("--entry-point", required=True, help="Entry-point function name.")
    push_p.add_argument("--description", default=None, help="Tool description.")
    push_p.add_argument("--requirements", default=None, help="Comma-separated pip requirements.")
    push_p.add_argument("--tags", default=None, help="Comma-separated tags.")
    push_p.add_argument("--timeout-ms", type=int, default=30000, help="Timeout in ms.")
    push_p.add_argument("--memory-mb", type=int, default=256, help="Memory limit in MB.")
    push_p.add_argument(
        "--scope", default="project", choices=["project", "global"],
        help="Tool scope.",
    )

    # -- register -----------------------------------------------------------
    reg_p = sub.add_parser("register", help="Register an external HTTP tool.")
    reg_p.add_argument("name", help="Tool name.")
    reg_p.add_argument("--endpoint-url", required=True, help="HTTP endpoint URL.")
    reg_p.add_argument("--description", default=None, help="Tool description.")
    reg_p.add_argument("--http-method", default="POST", help="HTTP method (default: POST).")
    reg_p.add_argument(
        "--auth-type", default="none",
        choices=["none", "api_key", "bearer", "oauth2"],
        help="Auth type.",
    )
    reg_p.add_argument("--tags", default=None, help="Comma-separated tags.")

    # -- invoke -------------------------------------------------------------
    inv_p = sub.add_parser("invoke", help="Invoke a tool by name.")
    inv_p.add_argument("tool_name", help="Name of the tool to invoke.")
    inv_p.add_argument("--input", default=None, help="JSON input as a string.")
    inv_p.add_argument("--input-file", default=None, help="Path to a JSON input file.")

    # -- list ---------------------------------------------------------------
    list_p = sub.add_parser("list", help="List tools.")
    list_p.add_argument("--tags", default=None, help="Comma-separated tag filter.")
    list_p.add_argument(
        "--type", default=None, choices=["hosted", "external", "mcp"],
        help="Filter by tool type.",
    )
    list_p.add_argument(
        "--status", default=None, choices=["active", "building", "build_failed"],
        help="Filter by status.",
    )

    # -- get ----------------------------------------------------------------
    get_p = sub.add_parser("get", help="Get a tool by ID.")
    get_p.add_argument("tool_id", help="Tool UUID.")

    # -- delete -------------------------------------------------------------
    del_p = sub.add_parser("delete", help="Delete a tool by ID.")
    del_p.add_argument("tool_id", help="Tool UUID.")

    # -- import-openapi ----------------------------------------------------
    io_p = sub.add_parser("import-openapi", help="Import tools from an OpenAPI spec.")
    io_p.add_argument("--namespace", required=True, help="Prefix for tool names (e.g. 'stripe').")
    io_p.add_argument("--spec-url", default=None, help="URL to fetch OpenAPI spec from.")
    io_p.add_argument("--spec-file", default=None, help="Local file path to OpenAPI spec JSON.")
    io_p.add_argument("--base-url", default=None, help="Override base URL for endpoints.")
    io_p.add_argument("--tags", default=None, help="Comma-separated tags.")
    io_p.add_argument(
        "--auth-type", default="none",
        choices=["none", "api_key", "bearer", "oauth2"],
        help="Auth type for imported tools.",
    )
    io_p.add_argument(
        "--scope", default="project", choices=["project", "global"],
        help="Scope for imported tools.",
    )

    # -- set-credential ----------------------------------------------------
    sc_p = sub.add_parser("set-credential", help="Store an encrypted credential for a tool.")
    sc_p.add_argument("--tool-id", required=True, help="Tool definition UUID.")
    sc_p.add_argument("--type", required=True, help="Credential type (e.g. api_key, oauth2).")
    sc_p.add_argument("--name", required=True, help="Human-readable credential name.")
    sc_p.add_argument("--payload", required=True, help="JSON string of secret key-value pairs.")
    sc_p.add_argument("--expires-at", default=None, help="ISO-8601 expiration timestamp.")

    # -- list-credentials --------------------------------------------------
    lc_p = sub.add_parser("list-credentials", help="List credentials for the current project.")
    lc_p.add_argument("--tool-id", default=None, help="Filter by tool definition UUID.")

    # -- revoke-credential -------------------------------------------------
    rc_p = sub.add_parser("revoke-credential", help="Revoke a credential.")
    rc_p.add_argument("--credential-id", required=True, help="Credential UUID.")
    rc_p.add_argument("--version", required=True, type=int, help="Credential version.")

    # -- mcp-register ------------------------------------------------------
    mr_p = sub.add_parser("mcp-register", help="Register an MCP server.")
    mr_p.add_argument("--name", required=True, help="Server name.")
    mr_p.add_argument("--transport", required=True, help="Transport type (e.g. sse, stdio).")
    mr_p.add_argument("--url", required=True, help="Server URL or connection string.")
    mr_p.add_argument("--description", default=None, help="Server description.")
    mr_p.add_argument("--tags", default=None, help="Comma-separated tags.")

    # -- mcp-list ----------------------------------------------------------
    ml_p = sub.add_parser("mcp-list", help="List MCP servers.")
    ml_p.add_argument("--status", default=None, help="Filter by health status.")
    ml_p.add_argument("--tags", default=None, help="Comma-separated tag filter.")

    # -- mcp-get -----------------------------------------------------------
    mg_p = sub.add_parser("mcp-get", help="Get an MCP server by ID.")
    mg_p.add_argument("--id", required=True, help="MCP server UUID.")

    # -- mcp-delete --------------------------------------------------------
    md_p = sub.add_parser("mcp-delete", help="Delete an MCP server.")
    md_p.add_argument("--id", required=True, help="MCP server UUID.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, Any] = {
    "push": _cmd_push,
    "register": _cmd_register,
    "invoke": _cmd_invoke,
    "list": _cmd_list,
    "get": _cmd_get,
    "delete": _cmd_delete,
    "import-openapi": _cmd_import_openapi,
    "set-credential": _cmd_set_credential,
    "list-credentials": _cmd_list_credentials,
    "revoke-credential": _cmd_revoke_credential,
    "mcp-register": _cmd_mcp_register,
    "mcp-list": _cmd_mcp_list,
    "mcp-get": _cmd_mcp_get,
    "mcp-delete": _cmd_mcp_delete,
}


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for ``taproot-tools``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except SystemExit:
        raise
    except ConnectionError as exc:
        _die(f"Connection error: {exc}")
    except Exception as exc:  # noqa: BLE001
        _die(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
