"""
scripts/api/routers/crates.py — Library crate and tag management endpoints.

Crates are named groups of songs (like DJ crates / playlists).
Tags and favorites are per-song annotations that persist across sessions.

Endpoints
---------
Crates:
  POST   /crates                      — create a crate
  GET    /crates                      — list all crates
  DELETE /crates/{crate_id}           — delete a crate
  PATCH  /crates/{crate_id}           — rename a crate
  GET    /crates/{crate_id}/songs     — list songs in a crate
  POST   /crates/{crate_id}/songs     — add a song to a crate
  DELETE /crates/{crate_id}/songs/{n} — remove a song from a crate

Tags:
  GET    /tags                        — all tags in use
  GET    /tags/{tag}/songs            — songs with a specific tag
  GET    /library/{name}/tags         — tags for a specific song
  POST   /library/{name}/tags         — add a tag to a song
  DELETE /library/{name}/tags/{tag}   — remove a tag from a song

Favorites:
  GET    /favorites                   — list favorited songs
  POST   /favorites/{name}            — favorite a song
  DELETE /favorites/{name}            — un-favorite a song
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from scripts.core import crates as crate_store

router = APIRouter()


# ---------------------------------------------------------------------------
# Crates
# ---------------------------------------------------------------------------

@router.post("/crates", status_code=201, tags=["crates"])
def create_crate(body: dict):
    """Create a new crate. Body: { "name": "..." }"""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")
    try:
        crate_id = crate_store.create_crate(name)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"crate_id": crate_id, "name": name}


@router.get("/crates", tags=["crates"])
def list_crates():
    """Return all crates with their song counts."""
    return {"crates": crate_store.list_crates()}


@router.delete("/crates/{crate_id}", tags=["crates"])
def delete_crate(crate_id: str):
    """Delete a crate and remove all its song memberships."""
    deleted = crate_store.delete_crate(crate_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Crate not found: {crate_id}")
    return JSONResponse({"crate_id": crate_id, "deleted": True})


@router.patch("/crates/{crate_id}", tags=["crates"])
def rename_crate(crate_id: str, body: dict):
    """Rename a crate. Body: { "name": "new name" }"""
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="'name' is required")
    updated = crate_store.rename_crate(crate_id, new_name)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Crate not found: {crate_id}")
    return {"crate_id": crate_id, "name": new_name}


@router.get("/crates/{crate_id}/songs", tags=["crates"])
def get_crate_songs(crate_id: str):
    """Return all songs in a crate, in the order they were added."""
    songs = crate_store.crate_songs(crate_id)
    return {"crate_id": crate_id, "songs": songs, "count": len(songs)}


@router.post("/crates/{crate_id}/songs", status_code=201, tags=["crates"])
def add_to_crate(crate_id: str, body: dict):
    """Add a song to a crate. Body: { "song_name": "..." }"""
    song_name = (body.get("song_name") or "").strip()
    if not song_name:
        raise HTTPException(status_code=400, detail="'song_name' is required")
    added = crate_store.add_to_crate(crate_id, song_name)
    if not added:
        raise HTTPException(status_code=409, detail=f"'{song_name}' is already in crate {crate_id}")
    return {"crate_id": crate_id, "song_name": song_name, "added": True}


@router.delete("/crates/{crate_id}/songs/{song_name:path}", tags=["crates"])
def remove_from_crate(crate_id: str, song_name: str):
    """Remove a song from a crate."""
    removed = crate_store.remove_from_crate(crate_id, song_name)
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"'{song_name}' is not in crate {crate_id}"
        )
    return JSONResponse({"crate_id": crate_id, "song_name": song_name, "removed": True})


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

@router.get("/tags", tags=["tags"])
def all_tags():
    """Return every distinct tag in use across the library."""
    return {"tags": crate_store.all_tags()}


@router.get("/tags/{tag}/songs", tags=["tags"])
def songs_by_tag(tag: str):
    """Return all songs that have a given tag."""
    songs = crate_store.songs_by_tag(tag)
    return {"tag": tag, "songs": songs, "count": len(songs)}


@router.get("/library/{song_name:path}/tags", tags=["tags"])
def get_song_tags(song_name: str):
    """Return all tags for a specific song."""
    return {"song_name": song_name, "tags": crate_store.song_tags(song_name)}


@router.post("/library/{song_name:path}/tags", status_code=201, tags=["tags"])
def add_song_tag(song_name: str, body: dict):
    """Add a tag to a song. Body: { "tag": "..." }"""
    tag = (body.get("tag") or "").strip()
    if not tag:
        raise HTTPException(status_code=400, detail="'tag' is required")
    added = crate_store.add_tag(song_name, tag)
    if not added:
        raise HTTPException(status_code=409, detail=f"Tag '{tag}' already exists on '{song_name}'")
    return {"song_name": song_name, "tag": tag, "added": True}


@router.delete("/library/{song_name:path}/tags/{tag}", tags=["tags"])
def remove_song_tag(song_name: str, tag: str):
    """Remove a tag from a song."""
    removed = crate_store.remove_tag(song_name, tag)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Tag '{tag}' not found on '{song_name}'")
    return JSONResponse({"song_name": song_name, "tag": tag, "removed": True})


# ---------------------------------------------------------------------------
# Favorites
# ---------------------------------------------------------------------------

@router.get("/favorites", tags=["favorites"])
def list_favorites():
    """Return all favorited song names, most recently favorited first."""
    songs = crate_store.list_favorites()
    return {"songs": songs, "count": len(songs)}


@router.post("/favorites/{song_name:path}", status_code=201, tags=["favorites"])
def add_favorite(song_name: str):
    """Mark a song as a favorite."""
    crate_store.set_favorite(song_name, value=True)
    return {"song_name": song_name, "favorited": True}


@router.delete("/favorites/{song_name:path}", tags=["favorites"])
def remove_favorite(song_name: str):
    """Remove a song from favorites."""
    if not crate_store.is_favorite(song_name):
        raise HTTPException(status_code=404, detail=f"'{song_name}' is not favorited")
    crate_store.set_favorite(song_name, value=False)
    return JSONResponse({"song_name": song_name, "favorited": False})
