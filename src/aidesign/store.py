"""Хранилище версий ProjectState: каждая версия — неизменяемый JSON-файл.

projects/<project_id>/v0001.json, v0002.json, ...
Для шага 0 этого достаточно; позже можно заменить на SQLite, не меняя интерфейс.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import ProjectState


class VersionStore:
    def __init__(self, root: str | Path = "projects"):
        self.root = Path(root)

    def _dir(self, project_id: str) -> Path:
        return self.root / project_id

    def save(self, state: ProjectState) -> Path:
        d = self._dir(state.project_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"v{state.version:04d}.json"
        if path.exists():
            raise FileExistsError(f"версия {state.version} уже сохранена: версии не перезаписываются")
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return path

    def versions(self, project_id: str) -> list[int]:
        return sorted(int(p.stem[1:]) for p in self._dir(project_id).glob("v*.json"))

    def load(self, project_id: str, version: int | None = None) -> ProjectState:
        vs = self.versions(project_id)
        if not vs:
            raise FileNotFoundError(f"нет сохранённых версий проекта {project_id}")
        v = version if version is not None else vs[-1]
        path = self._dir(project_id) / f"v{v:04d}.json"
        return ProjectState.model_validate_json(path.read_text(encoding="utf-8"))
