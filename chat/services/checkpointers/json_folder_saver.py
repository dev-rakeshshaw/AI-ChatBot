# chat/services/checkpointers/json_folder_saver.py
from __future__ import annotations

import asyncio
import base64
import json
import shutil
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Tuple, cast

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class JsonFolderSaver(BaseCheckpointSaver[str]):
    """
    Filesystem-backed checkpointer that stores checkpoints under:
        <root>/<thread_id>/_checkpoints/

    Files:
      - index.json                        # newest-first list of {"checkpoint_id","checkpoint_ns"}
      - <checkpoint_id>.json              # envelope: serialized checkpoint + metadata + versions
      - <checkpoint_id>.writes.ndjson     # optional per-write log (one JSON per line)

    Thread-safe within a single Python process (threading.Lock).
    For multi-process deployments, prefer SQLite/Postgres checkpointers.
    """

    def __init__(self, root: Path | str, *, serde: Optional[SerializerProtocol] = None) -> None:
        super().__init__(serde=serde)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonplus_serde = JsonPlusSerializer()
        self.lock = threading.Lock()

    # -------- internal paths --------
    def _thread_dir(self, thread_id: str, checkpoint_ns: str = "") -> Path:
        d = self.root / str(thread_id) / "_checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cp_path(self, thread_id: str, checkpoint_id: str, checkpoint_ns: str = "") -> Path:
        return self._thread_dir(thread_id, checkpoint_ns) / f"{checkpoint_id}.json"

    def _writes_path(self, thread_id: str, checkpoint_id: str, checkpoint_ns: str = "") -> Path:
        return self._thread_dir(thread_id, checkpoint_ns) / f"{checkpoint_id}.writes.ndjson"

    def _index_path(self, thread_id: str, checkpoint_ns: str = "") -> Path:
        return self._thread_dir(thread_id, checkpoint_ns) / "index.json"

    def _read_index(self, thread_id: str, checkpoint_ns: str = "") -> list[dict[str, Any]]:
        p = self._index_path(thread_id, checkpoint_ns)
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_index_entry(self, thread_id: str, checkpoint_ns: str, entry: dict[str, Any]) -> None:
        idx = self._read_index(thread_id, checkpoint_ns)
        idx.insert(0, entry)  # newest first
        self._index_path(thread_id, checkpoint_ns).write_text(
            json.dumps(idx, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -------- BaseCheckpointSaver: sync --------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self.lock:
            thread_id = str(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            cp_id = checkpoint["id"]

            type_, ser_bytes = self.serde.dumps_typed(checkpoint)
            merged_meta = get_checkpoint_metadata(config, metadata)
            meta_json = self.jsonplus_serde.dumps(merged_meta)

            envelope = {
                "type": type_,
                "checkpoint_base64": base64.b64encode(ser_bytes).decode("utf-8"),
                "metadata": json.loads(meta_json),
                "new_versions": new_versions,
            }

            cp_path = self._cp_path(thread_id, cp_id, checkpoint_ns)
            cp_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

            self._write_index_entry(
                thread_id,
                checkpoint_ns,
                {"checkpoint_id": cp_id, "checkpoint_ns": checkpoint_ns},
            )

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cp_id,
                }
            }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self.lock:
            thread_id = str(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = str(config["configurable"]["checkpoint_id"])

            wpath = self._writes_path(thread_id, checkpoint_id, checkpoint_ns)
            lines: list[str] = []
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                w_type, w_ser = self.serde.dumps_typed(value)
                lines.append(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "idx": write_idx,
                            "channel": channel,
                            "type": w_type,
                            "value_base64": base64.b64encode(w_ser).decode("utf-8"),
                        },
                        ensure_ascii=False,
                    )
                )

            with wpath.open("a", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln + "\n")

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        with self.lock:
            thread_id = str(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = get_checkpoint_id(config)

            if checkpoint_id:
                entry = {"checkpoint_id": checkpoint_id, "checkpoint_ns": checkpoint_ns}
            else:
                idx = self._read_index(thread_id, checkpoint_ns)
                if not idx:
                    return None
                entry = idx[0]
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": entry["checkpoint_id"],
                    }
                }

            cp_path = self._cp_path(thread_id, entry["checkpoint_id"], checkpoint_ns)
            if not cp_path.exists():
                return None

            envelope = json.loads(cp_path.read_text(encoding="utf-8"))
            type_ = envelope["type"]
            cp_bytes = base64.b64decode(envelope["checkpoint_base64"].encode("utf-8"))
            checkpoint = cast(Checkpoint, self.serde.loads_typed((type_, cp_bytes)))
            metadata = cast(CheckpointMetadata, envelope.get("metadata", {}))

            writes_file = self._writes_path(thread_id, entry["checkpoint_id"], checkpoint_ns)
            writes_list: list[Tuple[str, str, Any]] = []
            if writes_file.exists():
                with writes_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            o = json.loads(line)
                            w_val = self.serde.loads_typed(
                                (o["type"], base64.b64decode(o["value_base64"]))
                            )
                            writes_list.append((o["task_id"], o["channel"], w_val))
                        except Exception:
                            continue

            return CheckpointTuple(
                config,
                checkpoint,
                metadata,
                None,  # no parent tracking
                writes_list,
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None or "configurable" not in config or "thread_id" not in config["configurable"]:
            return iter(())

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        idx = self._read_index(thread_id, checkpoint_ns)

        stop_id = get_checkpoint_id(before) if before and get_checkpoint_id(before) else None
        count = 0

        for entry in idx:
            if stop_id and entry["checkpoint_id"] == stop_id:
                break
            cp_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": entry["checkpoint_id"],
                }
            }
            tup = self.get_tuple(cp_config)
            if tup:
                yield tup
                count += 1
                if limit and count >= limit:
                    break

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        return f"{next_v:032}"

    def delete_thread(self, thread_id: str) -> None:
        with self.lock:
            tdir = self.root / str(thread_id)
            if tdir.exists():
                shutil.rmtree(tdir, ignore_errors=True)

    # -------- async wrappers --------
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        tuples = list(self.list(config, filter=filter, before=before, limit=limit))
        for t in tuples:
            yield t







