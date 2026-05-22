import unittest
import os
import xml.etree.ElementTree as ET
from autodj.utils import export_rekordbox_xml

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.test_xml = "test_rekordbox.xml"
        self.mock_tracklist = [
            {
                'file': 'track1.flac',
                'genre': 'Techno',
                'key': '1A',
                'bpm': '128',
                'start_ms': 0,
                'duration_ms': 300000,
                'path': '/app/tracks/track1.flac'
            },
            {
                'file': 'track2.flac',
                'genre': 'House',
                'key': '2A',
                'bpm': '124',
                'start_ms': 280000,
                'duration_ms': 320000,
                'path': '/app/tracks/track2.flac'
            }
        ]

    def tearDown(self):
        if os.path.exists(self.test_xml):
            os.remove(self.test_xml)

    def test_rekordbox_xml_generation(self):
        """Validates that the Rekordbox XML structure is correct."""
        export_rekordbox_xml(self.mock_tracklist, self.test_xml)

        self.assertTrue(os.path.exists(self.test_xml))

        tree = ET.parse(self.test_xml)
        root = tree.getroot()

        self.assertEqual(root.tag, "DJ_PLAYLISTS")

        collection = root.find("COLLECTION")
        self.assertIsNotNone(collection)
        self.assertEqual(collection.get("Entries"), "2")

        tracks = collection.findall("TRACK")
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].get("Name"), "track1.flac")
        self.assertEqual(tracks[0].get("Tonality"), "1A")

        # Check for Cue point
        cue = tracks[1].find("POSITION_MARK")
        self.assertIsNotNone(cue)
        self.assertEqual(cue.get("Name"), "Transition Start")
        self.assertEqual(cue.get("Start"), "280.0")

if __name__ == "__main__":
    unittest.main()
