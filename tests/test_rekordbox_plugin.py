
import os
import pytest
from autodj.plugins import RekordboxSourcePlugin

def test_rekordbox_plugin():
    # Create a dummy track file
    track_path = os.path.abspath("test_track_rb.flac")
    with open(track_path, "wb") as f:
        f.write(b"fakedata")

    # Create a sample Rekordbox XML
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <DJ_PLAYLISTS Version="1.0.0">
      <COLLECTION Entries="1">
        <TRACK TrackID="1" Location="file://localhost{track_path.replace(os.sep, '/')}">
        </TRACK>
      </COLLECTION>
    </DJ_PLAYLISTS>
    """
    xml_path = "test_rekordbox_rb.xml"
    with open(xml_path, "w") as f:
        f.write(xml_content)

    try:
        plugin = RekordboxSourcePlugin()
        tracks = plugin.get_tracks(xml_path=xml_path)

        assert len(tracks) == 1
        assert tracks[0] == track_path
    finally:
        if os.path.exists(track_path):
            os.remove(track_path)
        if os.path.exists(xml_path):
            os.remove(xml_path)
