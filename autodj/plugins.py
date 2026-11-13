"""Modular Plugin System | Auto DJ Script.
==============================================
Provides extensible hooks for:
- `SourcePlugin`: Track discovery (`local_folder`, `rekordbox_xml`, custom `plugins/*.py`).
- `OutputPlugin`: Mix delivery (`local_file`, `icecast_stream`, custom `plugins/*.py`).
- `ToolPlugin`: Lifecycle event hooks (`pre_mix`, `on_track_start`, `post_mix`).
"""

import glob
import importlib.util
import os
import subprocess
import tempfile
from typing import Dict, List, Type


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
        raise NotImplementedError


class OutputPlugin(BasePlugin):
    """Plugins that handle the final mix output (export, stream, etc.)."""

    def export(self, master_audio, tracklist, enriched_metadata=None, **kwargs):
        raise NotImplementedError


class ToolPlugin(BasePlugin):
    """Plugins that provide utility hooks during the mixing process."""

    def pre_mix(self, status_obj=None, **kwargs):
        pass

    def post_mix(self, status_obj=None, **kwargs):
        pass

    def on_track_start(self, track_meta, status_obj=None, **kwargs):
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
        """Dynamically loads external `.py` plugins from `plugins_dir`."""
        if not os.path.exists(plugins_dir):
            try:
                os.makedirs(plugins_dir, exist_ok=True)
            except Exception:
                pass
            return

        import sys

        if plugins_dir not in sys.path:
            sys.path.append(plugins_dir)

        for filename in sorted(os.listdir(plugins_dir)):
            if filename.endswith(".py") and filename != "__init__.py":
                plugin_path = os.path.join(plugins_dir, filename)
                module_name = filename[:-3]

                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, plugin_path
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        print(f"[*] Modular Plugin System: Loaded {filename}")
                except Exception as e:
                    print(f"[!] Failed to load plugin {filename}: {e}")


@PluginRegistry.register_output
class LocalFileSink(OutputPlugin):
    """Default output plugin that saves lossless FLAC, tracklist TXT, Rekordbox XML, and optional MP3."""

    name = "local_file"
    display_name = "Local Lossless FLAC (+ XML/TXT)"
    description = (
        "Exports the final mix to a local FLAC file, generates timestamped tracklist & Rekordbox XML cues."
    )

    def export(self, master_audio, tracklist, enriched_metadata=None, **kwargs):
        output_path = kwargs.get("output")
        version = kwargs.get("version", "Unknown")
        all_files = kwargs.get("all_files", [])
        meta_list = kwargs.get("meta_list", [])
        processed_tracks = kwargs.get("processed_tracks", [])
        export_mp3 = bool(kwargs.get("export_mp3", False))

        if not output_path:
            return

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        duration_ms = len(master_audio)
        sample_count = int(duration_ms * master_audio.frame_rate / 1000)
        estimated_bytes = sample_count * 2 * 2
        max_wav_size = 3 * 1024 * 1024 * 1024  # 3 GB WAV pipe threshold

        use_chunked_export = estimated_bytes > max_wav_size

        print(
            f"[*] Exporting {duration_ms / 1000 / 60:.1f} min mix to {output_path}..."
        )

        if use_chunked_export:
            print(
                "  [LARGE] Using soundfile direct FLAC writer to bypass 4 GB WAV pipe limit"
            )
            self._export_soundfile(master_audio, output_path)
        else:
            master_audio.export(output_path, format="flac")
            print("[*] Export complete!")

        # Standard Tracklist Export
        tl_path = os.path.splitext(output_path)[0] + "_tracklist.txt"
        with open(tl_path, "w", encoding="utf-8") as f:
            f.write(f"Auto DJ v{version} Master Tracklist\n{'=' * 40}\n")
            for item in tracklist:
                f.write(
                    f"[{item['timestamp']}] {item['file']} ({item['key']}) [{item['genre']}]\n"
                )

        # Integration Bridge: Rekordbox XML Export with 3-point transition cues
        from .utils import export_rekordbox_xml, transcode_to_mp3

        xml_path = os.path.splitext(output_path)[0] + "_rekordbox.xml"
        try:
            enriched_tl = []
            for i, item in enumerate(tracklist):
                entry = dict(item)
                entry["path"] = all_files[i] if i < len(all_files) else ""
                entry["bpm"] = str(meta_list[i]["bpm"]) if i < len(meta_list) else "0"
                pt = processed_tracks[i] if i < len(processed_tracks) else None
                entry["duration_ms"] = (
                    len(pt[0]) if (pt is not None and pt[0] is not None) else 0
                )
                enriched_tl.append(entry)

            export_rekordbox_xml(enriched_tl, xml_path)
            print(f"[*] Integration: Rekordbox XML exported to {xml_path}")
        except Exception as e:
            print(f"[WARN] Rekordbox export failed: {e}")

        if export_mp3:
            try:
                mp3_out = transcode_to_mp3(output_path)
                print(f"[*] Transcoded 320kbps MP3 copy to {mp3_out}")
            except Exception as e:
                print(f"[WARN] MP3 transcode failed: {e}")

    def _export_soundfile(self, master_audio, output_path):
        """Write FLAC using `soundfile` directly to handle multi-hour >4 GB renders."""
        import numpy as np
        import soundfile as sf

        samples = np.array(master_audio.get_array_of_samples(), dtype=np.int16)
        if master_audio.channels == 2:
            samples = samples.reshape(-1, 2)
        else:
            samples = samples.reshape(-1, 1)

        sf.write(
            output_path,
            samples,
            master_audio.frame_rate,
            format="FLAC",
            subtype="PCM_16",
        )
        size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"[*] Export complete! ({size_gb:.2f} GB)")

    def _export_chunked(self, master_audio, output_path, version, tracklist):
        """Fallback chunked FFmpeg FLAC concatenation."""
        chunk_ms = 60 * 30 * 1000
        total_ms = len(master_audio)
        chunk_files = []

        with tempfile.TemporaryDirectory() as tmpdir:
            start_ms = 0
            chunk_idx = 0
            while start_ms < total_ms:
                end_ms = min(start_ms + chunk_ms, total_ms)
                chunk = master_audio[start_ms:end_ms]
                chunk_path = os.path.join(tmpdir, f"chunk_{chunk_idx:03d}.flac")
                chunk.export(chunk_path, format="flac")
                chunk_files.append(chunk_path)
                start_ms = end_ms
                chunk_idx += 1

            file_list = os.path.join(tmpdir, "chunks.txt")
            with open(file_list, "w", encoding="utf-8") as f:
                for cf in chunk_files:
                    posix_path = cf.replace("\\", "/")
                    f.write(f"file '{posix_path}'\n")

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    file_list,
                    "-c:a",
                    "flac",
                    output_path,
                ],
                capture_output=True,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode() if result.stderr else "unknown"
                raise RuntimeError(f"Failed to concatenate audio chunks: {stderr}")


@PluginRegistry.register_output
class IcecastStreamSink(OutputPlugin):
    """Streams the rendered master mix to an Icecast / Shoutcast / RTMP endpoint via FFmpeg."""

    name = "icecast_stream"
    display_name = "Icecast / RTMP Live Broadcast"
    description = "Broadcasts real-time encoded audio to an Icecast2, Shoutcast, or RTMP URL via FFmpeg."

    def export(self, master_audio, tracklist, enriched_metadata=None, **kwargs):
        stream_url = kwargs.get("stream_url")
        output_path = kwargs.get("output")

        # Always preserve local lossless backup first when output_path is provided
        if output_path:
            LocalFileSink().export(
                master_audio, tracklist, enriched_metadata=enriched_metadata, **kwargs
            )

        if not stream_url:
            print("[WARN] IcecastStreamSink: No stream_url provided; skipped live push.")
            return

        src_file = output_path
        tmp_ctx = None
        if not src_file or not os.path.exists(src_file):
            tmp_ctx = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
            tmp_ctx.close()
            master_audio.export(tmp_ctx.name, format="flac")
            src_file = tmp_ctx.name

        try:
            fmt = "flv" if str(stream_url).startswith("rtmp://") else "mp3"
            cmd = [
                "ffmpeg",
                "-re",
                "-i",
                src_file,
                "-vn",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "320k",
                "-f",
                fmt,
                str(stream_url),
            ]
            print(f"[*] Broadcasting master stream to {stream_url}...")
            proc = subprocess.run(cmd, capture_output=True, timeout=15)
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode(errors="replace")[-300:]
                print(f"[WARN] Live broadcast exited ({proc.returncode}): {err}")
        except Exception as e:
            print(f"[WARN] IcecastStreamSink broadcast warning: {e}")
        finally:
            if tmp_ctx and os.path.exists(tmp_ctx.name):
                try:
                    os.remove(tmp_ctx.name)
                except OSError:
                    pass


@PluginRegistry.register_source
class LocalFolderSource(SourcePlugin):
    """Default source plugin that scans a local directory for audio files."""

    name = "local_folder"
    display_name = "Local Folder"
    description = "Scans a local directory for supported lossless and compressed audio formats."

    def get_tracks(self, **kwargs) -> List[str]:
        folder = kwargs.get("folder")
        extensions = kwargs.get("extensions", [".flac", ".wav", ".mp3", ".aiff", ".m4a"])
        if not folder or not os.path.exists(folder):
            return []

        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))

        # Case-normalized deduplication on Windows while preserving deterministic alphabetical order
        seen = {}
        for f in sorted(os.path.abspath(p) for p in files):
            norm_key = os.path.normcase(f)
            if norm_key not in seen:
                seen[norm_key] = f
        return list(seen.values())


@PluginRegistry.register_source
class RekordboxSourcePlugin(SourcePlugin):
    """Source plugin that reads tracks from a Pioneer Rekordbox XML export (`pioneer.xml`)."""

    name = "rekordbox_xml"
    display_name = "Rekordbox XML"
    description = "Imports analyzed tracks and playlist ordering from a Rekordbox XML collection."

    def get_tracks(self, **kwargs) -> List[str]:
        xml_path = kwargs.get("xml_path")
        folder = kwargs.get("folder")
        if (not xml_path or not os.path.exists(str(xml_path))) and folder and str(folder).lower().endswith(".xml"):
            xml_path = folder
        if not xml_path or not os.path.exists(str(xml_path)):
            return []

        import urllib.parse
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            tracks = []
            for track in root.findall(".//TRACK"):
                location = track.get("Location")
                if location:
                    path = location.replace("file://localhost", "")
                    path = urllib.parse.unquote(path)

                    if os.name == "nt" and path.startswith("/") and ":" in path:
                        path = path[1:]

                    if os.path.exists(path):
                        tracks.append(os.path.abspath(path))

            return tracks
        except Exception as e:
            print(f"[!] RekordboxSourcePlugin failed: {e}")
            return []
