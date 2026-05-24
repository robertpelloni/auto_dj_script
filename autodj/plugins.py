"""
Modular Plugin System | Auto DJ Script (7.7.0)
==============================================
This module provides the architectural foundation for extending Auto DJ
with diverse input sources, output sinks, and third-party tools.

Architecture:
- SourcePlugin: For track discovery (Local Folder, Spotify, S3, etc.)
- OutputPlugin: For mix delivery (Local File, Icecast, SoundCloud, etc.)
- ToolPlugin: For hook-based utility integration.
"""

import os
import glob
import importlib.util
from typing import Dict, List, Type, Any

class BasePlugin:
    """Base class for all Auto DJ plugins."""
    name = "base_plugin"
    display_name = "Base Plugin"
    description = ""
    version = "1.0.0"
    author = "Auto DJ"

class SourcePlugin(BasePlugin):
    """Plugins that provide audio tracks for the mixing engine."""
    def get_tracks(self, **kwargs) -> List[str]:
        """Returns a list of file paths or identifiers for tracks."""
        raise NotImplementedError

class OutputPlugin(BasePlugin):
    """Plugins that handle the final mix output (export, stream, etc.)."""
    def export(self, master_audio, tracklist, enriched_metadata=None, **kwargs):
        """Processes the final master audio and tracklist."""
        raise NotImplementedError

class ToolPlugin(BasePlugin):
    """Plugins that provide utility hooks during the mixing process."""
    def pre_mix(self, status_obj=None, **kwargs):
        """Hook called before the mixing loop starts."""
        pass

    def post_mix(self, status_obj=None, **kwargs):
        """Hook called after the mixing loop completes."""
        pass

    def on_track_start(self, track_meta, status_obj=None, **kwargs):
        """Hook called when a new track starts in the mix."""
        pass

class PluginRegistry:
    _sources: Dict[str, Type[SourcePlugin]] = {}
    _outputs: Dict[str, Type[OutputPlugin]] = {}
    _tools: Dict[str, Type[ToolPlugin]] = {}

    @classmethod
    def register_source(cls, plugin_cls: Type[SourcePlugin]):
        cls._sources[plugin_cls.name] = plugin_cls
        return plugin_cls

    @classmethod
    def register_output(cls, plugin_cls: Type[OutputPlugin]):
        cls._outputs[plugin_cls.name] = plugin_cls
        return plugin_cls

    @classmethod
    def register_tool(cls, plugin_cls: Type[ToolPlugin]):
        cls._tools[plugin_cls.name] = plugin_cls
        return plugin_cls

    @classmethod
    def get_sources(cls) -> Dict[str, Type[SourcePlugin]]:
        return cls._sources

    @classmethod
    def get_outputs(cls) -> Dict[str, Type[OutputPlugin]]:
        return cls._outputs

    @classmethod
    def get_tools(cls) -> Dict[str, Type[ToolPlugin]]:
        return cls._tools

    @classmethod
    def load_plugins(cls, plugins_dir: str):
        """Dynamically loads plugins from the specified directory."""
        if not os.path.exists(plugins_dir):
            try:
                os.makedirs(plugins_dir, exist_ok=True)
            except Exception:
                pass
            return

        import sys
        if plugins_dir not in sys.path:
            sys.path.append(plugins_dir)

        for filename in os.listdir(plugins_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                plugin_path = os.path.join(plugins_dir, filename)
                module_name = filename[:-3]

                try:
                    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        print(f"[*] Modular Plugin System: Loaded {filename}")
                except Exception as e:
                    print(f"[!] Failed to load plugin {filename}: {e}")

@PluginRegistry.register_output
class LocalFileSink(OutputPlugin):
    """Default output plugin that saves the mix to a local file and tracklist."""
    name = "local_file"
    display_name = "Local File"
    description = "Exports the final mix to a local FLAC file and generates a tracklist."

    def export(self, master_audio, tracklist, enriched_metadata=None, **kwargs):
        output_path = kwargs.get('output')
        version = kwargs.get('version', 'Unknown')
        all_files = kwargs.get('all_files', [])
        meta_list = kwargs.get('meta_list', [])
        processed_tracks = kwargs.get('processed_tracks', [])

        if not output_path:
            return

        print(f"[*] Exporting {len(master_audio)/1000/60:.1f} min mix to {output_path}...")
        master_audio.export(output_path, format="flac")
        print(f"[*] Export complete!")

        # Standard Tracklist Export
        tl_path = os.path.splitext(output_path)[0] + "_tracklist.txt"
        with open(tl_path, "w") as f:
            f.write(f"Auto DJ v{version} Master Tracklist\n{'='*40}\n")
            for item in tracklist:
                f.write(f"[{item['timestamp']}] {item['file']} ({item['key']}) [{item['genre']}]\n")

        # Integration Bridge: Rekordbox XML Export
        from .utils import export_rekordbox_xml
        xml_path = os.path.splitext(output_path)[0] + "_rekordbox.xml"
        try:
            enriched_tl = []
            for i, item in enumerate(tracklist):
                entry = dict(item)
                entry['path'] = all_files[i] if i < len(all_files) else ""
                entry['bpm'] = str(meta_list[i]['bpm']) if i < len(meta_list) else "0"
                entry['duration_ms'] = len(processed_tracks[i][0]) if i < len(processed_tracks) else 0
                enriched_tl.append(entry)

            export_rekordbox_xml(enriched_tl, xml_path)
            print(f"[*] Integration: Rekordbox XML exported to {xml_path}")
        except Exception as e:
            print(f"[WARN] Rekordbox export failed: {e}")

@PluginRegistry.register_source
class LocalFolderSource(SourcePlugin):
    """Default source plugin that scans a local directory for audio files."""
    name = "local_folder"
    display_name = "Local Folder"
    description = "Scans a local directory for supported audio formats."

    def get_tracks(self, **kwargs) -> List[str]:
        folder = kwargs.get('folder')
        extensions = kwargs.get('extensions', ['.flac', '.wav', '.mp3'])
        if not folder or not os.path.exists(folder):
            return []

        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))

        return list(set(os.path.abspath(f) for f in files))

@PluginRegistry.register_source
class RekordboxSourcePlugin(SourcePlugin):
    """
    Source plugin that reads from a Rekordbox XML export (pioneer.xml).
    Enables importing analyzed tracks, hot cues, and playlist structures.
    """
    name = "rekordbox_xml"
    display_name = "Rekordbox XML"
    description = "Imports tracks directly from a Rekordbox pioneer.xml export."

    def get_tracks(self, **kwargs) -> List[str]:
        xml_path = kwargs.get('xml_path')
        if not xml_path or not os.path.exists(xml_path):
            return []

        import xml.etree.ElementTree as ET
        import urllib.parse

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            tracks = []
            for track in root.findall(".//TRACK"):
                location = track.get("Location")
                if location:
                    # Handle file://localhost/... format
                    path = location.replace("file://localhost", "")
                    path = urllib.parse.unquote(path)

                    # On Windows, path might start with /C:/
                    if os.name == 'nt' and path.startswith("/") and ":" in path:
                        path = path[1:]

                    if os.path.exists(path):
                        tracks.append(os.path.abspath(path))

            return tracks
        except Exception as e:
            print(f"[!] RekordboxSourcePlugin failed: {e}")
            return []
