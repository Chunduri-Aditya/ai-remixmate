"""
Route-shadowing regression tests.

FastAPI resolves routes in registration order, so a static path like
GET /library/crate-search silently loses to a dynamic GET /library/{name}
registered before it — the request reaches the song-lookup handler with
name="crate-search" and 404s. That exact bug shipped once (analysis.router
was included after library.router in routes.py); these tests pin the fix.

Pure route-table inspection — no TestClient, no jobs, no audio.
"""

from fastapi.routing import APIRoute, RouteContext, iter_route_contexts

from scripts.api.routes import router


def _api_routes() -> list[RouteContext]:
    """The flattened route table in dispatch order. FastAPI >= 0.141 keeps
    `include_router` lazy — `router.routes` holds `_IncludedRouter` entries,
    not `APIRoute`s — so walk it the way FastAPI's own OpenAPI generator does."""
    return [
        rc for rc in iter_route_contexts(router.routes)
        if isinstance(rc.original_route, APIRoute)
    ]


def _first_matching_route(path: str, method: str) -> APIRoute:
    """Return the first route whose regex matches `path` for `method` —
    i.e. the one FastAPI would actually dispatch to."""
    for rc in _api_routes():
        if method not in rc.methods:
            continue
        if rc.path_regex.match(path):
            return rc.original_route
    raise AssertionError(f"no route matches {method} {path}")


class TestStaticRoutesWinOverDynamic:
    def test_crate_search_not_shadowed_by_song_lookup(self):
        route = _first_matching_route("/library/crate-search", "GET")
        assert route.path == "/library/crate-search", (
            f"GET /library/crate-search dispatches to {route.path!r} — "
            "analysis.router must be included before library.router in routes.py"
        )

    def test_library_names_not_shadowed(self):
        route = _first_matching_route("/library/names", "GET")
        assert route.path == "/library/names"

    def test_processing_status_not_shadowed(self):
        route = _first_matching_route("/library/processing-status", "GET")
        assert route.path == "/library/processing-status"

    def test_storage_not_shadowed(self):
        route = _first_matching_route("/library/storage", "GET")
        assert route.path == "/library/storage"

    def test_song_lookup_still_reachable(self):
        route = _first_matching_route("/library/Some%20Song", "GET")
        assert route.path == "/library/{name}"

    def test_scan_broken_not_shadowed(self):
        route = _first_matching_route("/library/scan-broken", "POST")
        assert route.path == "/library/scan-broken"

    def test_delete_batch_not_shadowed(self):
        route = _first_matching_route("/library/delete-batch", "POST")
        assert route.path == "/library/delete-batch"

    def test_structure_v1_endpoint_reachable(self):
        """DJ_ENGINE_IMPROVEMENT_PLAN.md §3.1 — GET /analysis/v1/{name}/structure
        lives entirely under a prefix ('/analysis/v1/...') no other router
        uses, so nothing before it in routes.py can shadow it — but pin that
        down explicitly rather than relying on the generic static-route guard
        below, which only checks parameter-free paths."""
        route = _first_matching_route("/analysis/v1/Some%20Song/structure", "GET")
        assert route.path == "/analysis/v1/{name}/structure"

    def test_every_static_route_wins_over_dynamic_siblings(self):
        """Generic guard: for every static (no-param) path and method in the
        app, the route FastAPI dispatches to must be that same static route."""
        static_routes = [
            (rc.path, method)
            for rc in _api_routes()
            if "{" not in rc.path
            for method in rc.methods
        ]
        misrouted = []
        for path, method in static_routes:
            winner = _first_matching_route(path, method)
            if winner.path != path:
                misrouted.append((method, path, winner.path))
        assert not misrouted, f"static routes shadowed by earlier dynamic routes: {misrouted}"
