#!/usr/bin/env python3
"""
Port selection.

Every entry point asks for a preferred port and gets the first one that is
actually free, so a stale server, another tool, or a second copy of the suite
can never stop you from starting up.
"""
import errno
import os
import socket

# How far past the preferred port to walk before giving up and letting the OS
# hand out an ephemeral one. Wide enough to step over a row of dead servers,
# narrow enough that the port you land on is still recognisably "yours".
SEARCH_SPAN = 100


def is_free(port, host="0.0.0.0"):
    """True if `host:port` can be bound right now.

    Probes with the same host the caller will serve on: binding 0.0.0.0 fails
    when something holds 127.0.0.1 on that port, so probing a different address
    than you serve on would give the wrong answer.
    """
    if not 0 < port < 65536:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # SO_REUSEADDR matches what the server sets, so a port left in TIME_WAIT
        # by a just-stopped server reads as free — it is. A *live* listener still
        # fails to bind, which is exactly the case we want to detect.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES, errno.EADDRNOTAVAIL):
                return False
            raise
        return True


def find_free(preferred, host="0.0.0.0", span=SEARCH_SPAN):
    """Return `preferred` if it is free, otherwise the next free port above it.

    Falls back to an OS-assigned ephemeral port if the whole span is occupied —
    on a machine that busy, *some* port beats refusing to start.
    """
    for port in range(preferred, min(preferred + span, 65536)):
        if is_free(port, host):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def resolve(preferred, host="0.0.0.0", env_var=None, announce=True):
    """Pick the port to serve on, and say so if it isn't the one asked for.

    `env_var` keeps the choice stable across Werkzeug's auto-reloader. Under the
    reloader the *parent* binds the listening socket and hands the file
    descriptor to each restarted child, so the child must not go looking for a
    port of its own: probing would find the parent's socket, report "busy", and
    drift one port higher on every reload — while the server carried on serving
    on the inherited fd, leaving the printed URL wrong. So once the port is in
    the environment, an already-resolved answer is what it is: take it as final.
    """
    if env_var:
        inherited = os.environ.get(env_var)
        if inherited and inherited.isdigit():
            return int(inherited)

    port = find_free(preferred, host)
    if announce and port != preferred:
        print(f"  Port {preferred} is in use — using {port} instead.")
    if env_var:
        os.environ[env_var] = str(port)
    return port
