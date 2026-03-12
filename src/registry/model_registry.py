"""Simple local model registry abstraction with deployment stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import ModelRegistryError
from src.utils.io import load_json, save_json

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
ARCHIVED_ALIAS = "archived"
PRODUCTION_STAGE = "production"
STAGING_STAGE = "staging"
ARCHIVED_STAGE = "archived"


@dataclass(slots=True)
class RegistryEntry:
    """Metadata for a registered model."""

    model_name: str
    version: str
    version_number: int
    stage: str
    serving_alias: str
    artifact_path: str
    run_id: str
    metrics: dict[str, float]
    created_at: str
    validation_issues: list[str]
    dataset_version: str
    dataset_fingerprint: str


@dataclass(slots=True)
class RegistrySummary:
    """Summary of active champion and challenger versions."""

    champion: RegistryEntry | None
    challenger: RegistryEntry | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to a JSON-compatible structure."""
        return {
            "champion": asdict(self.champion) if self.champion else None,
            "challenger": asdict(self.challenger) if self.challenger else None,
        }


class LocalModelRegistry:
    """JSON-backed registry for local development and demos."""

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.registry_dir / "registry.json"

    def _load_payload(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"models": [], "aliases": {}, "history": []}
        payload = load_json(self.registry_path)
        payload.setdefault("models", [])
        payload.setdefault("aliases", {})
        payload.setdefault("history", [])
        return payload

    def _save_payload(self, payload: dict[str, Any]) -> None:
        save_json(payload, self.registry_path)

    @staticmethod
    def _record_history(payload: dict[str, Any], model_name: str, alias: str, version: str) -> None:
        payload["history"].append(
            {
                "model_name": model_name,
                "alias": alias,
                "version": version,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @staticmethod
    def _update_entry_alias(entry: dict[str, Any], alias: str) -> None:
        entry["serving_alias"] = alias
        if alias == CHAMPION_ALIAS:
            entry["stage"] = PRODUCTION_STAGE
        elif alias == CHALLENGER_ALIAS:
            entry["stage"] = STAGING_STAGE
        else:
            entry["stage"] = ARCHIVED_STAGE

    def _get_entries(self, payload: dict[str, Any], model_name: str) -> list[RegistryEntry]:
        return [RegistryEntry(**item) for item in payload["models"] if item["model_name"] == model_name]

    def get_next_version_number(self, model_name: str) -> int:
        """Return the next monotonically increasing version number for a model line."""
        payload = self._load_payload()
        entries = self._get_entries(payload, model_name)
        if not entries:
            return 1
        return max(entry.version_number for entry in entries) + 1

    @staticmethod
    def build_version_label(model_name: str, version_number: int) -> str:
        """Create a lightweight version label used across registry and API."""
        normalized_name = model_name.replace("_", "-")
        return f"{normalized_name}-v{version_number:04d}"

    def _get_entry_by_version(
        self,
        payload: dict[str, Any],
        model_name: str,
        version: str,
    ) -> RegistryEntry:
        matches = [
            RegistryEntry(**item)
            for item in payload["models"]
            if item["model_name"] == model_name and item["version"] == version
        ]
        if not matches:
            raise ModelRegistryError(
                f"No registry entry found for model={model_name!r}, version={version!r}",
                details={"model_name": model_name, "version": version},
            )
        return sorted(matches, key=lambda item: item.created_at, reverse=True)[0]

    def _assign_alias(
        self,
        payload: dict[str, Any],
        model_name: str,
        alias: str,
        version: str,
    ) -> None:
        aliases = payload["aliases"].setdefault(model_name, {})
        previous_version = aliases.get(alias)

        if alias == CHAMPION_ALIAS and aliases.get(CHALLENGER_ALIAS) == version:
            aliases.pop(CHALLENGER_ALIAS, None)
        elif alias == CHALLENGER_ALIAS and aliases.get(CHAMPION_ALIAS) == version:
            aliases.pop(CHAMPION_ALIAS, None)

        aliases[alias] = version

        for entry in payload["models"]:
            if entry["model_name"] != model_name:
                continue
            if entry["version"] == version:
                self._update_entry_alias(entry, alias)
            elif previous_version and entry["version"] == previous_version:
                self._update_entry_alias(entry, ARCHIVED_ALIAS)

        self._record_history(payload, model_name=model_name, alias=alias, version=version)

    def register_model(
        self,
        model_name: str,
        version: str,
        stage: str,
        artifact_path: Path,
        run_id: str,
        metrics: dict[str, float],
        validation_issues: list[str],
        version_number: int,
        dataset_version: str,
        dataset_fingerprint: str,
    ) -> RegistryEntry:
        """Register a model version and persist metadata to disk."""
        payload = self._load_payload()
        now = datetime.now(timezone.utc).isoformat()

        champion_exists = model_name in payload["aliases"] and CHAMPION_ALIAS in payload["aliases"][model_name]
        serving_alias = CHALLENGER_ALIAS if validation_issues or champion_exists else CHAMPION_ALIAS
        resolved_stage = STAGING_STAGE if serving_alias == CHALLENGER_ALIAS else stage

        entry = RegistryEntry(
            model_name=model_name,
            version=version,
            version_number=version_number,
            stage=resolved_stage,
            serving_alias=serving_alias,
            artifact_path=str(artifact_path),
            run_id=run_id,
            metrics=metrics,
            created_at=now,
            validation_issues=validation_issues,
            dataset_version=dataset_version,
            dataset_fingerprint=dataset_fingerprint,
        )

        payload["models"].append(asdict(entry))

        if serving_alias == CHAMPION_ALIAS:
            self._assign_alias(payload, model_name=model_name, alias=CHAMPION_ALIAS, version=version)
        else:
            self._assign_alias(payload, model_name=model_name, alias=CHALLENGER_ALIAS, version=version)

        self._save_payload(payload)
        return entry

    def get_model(self, model_name: str, stage: str = "production") -> RegistryEntry:
        """Get the latest model entry for a given stage."""
        payload = self._load_payload()
        candidates = [
            RegistryEntry(**item)
            for item in payload["models"]
            if item["model_name"] == model_name and item["stage"].lower() == stage.lower()
        ]
        if not candidates:
            raise ModelRegistryError(
                f"No model found for model_name={model_name!r} and stage={stage!r}",
                details={"model_name": model_name, "stage": stage},
            )
        candidates.sort(key=lambda item: item.created_at, reverse=True)
        return candidates[0]

    def get_model_by_alias(self, model_name: str, alias: str = "champion") -> RegistryEntry:
        """Resolve a serving alias such as champion or challenger."""
        payload = self._load_payload()
        aliases = payload["aliases"].get(model_name, {})
        version = aliases.get(alias)
        if not version:
            raise ModelRegistryError(
                f"No model found for model_name={model_name!r} and alias={alias!r}",
                details={"model_name": model_name, "alias": alias},
            )
        return self._get_entry_by_version(payload, model_name=model_name, version=version)

    def get_registry_summary(self, model_name: str) -> RegistrySummary:
        """Return active champion and challenger entries."""
        payload = self._load_payload()
        aliases = payload["aliases"].get(model_name, {})
        champion = None
        challenger = None
        if aliases.get(CHAMPION_ALIAS):
            champion = self._get_entry_by_version(payload, model_name, aliases[CHAMPION_ALIAS])
        if aliases.get(CHALLENGER_ALIAS):
            challenger = self._get_entry_by_version(payload, model_name, aliases[CHALLENGER_ALIAS])
        return RegistrySummary(champion=champion, challenger=challenger)

    def promote_model_version(self, model_name: str, version: str) -> RegistryEntry:
        """Promote a specific version to champion and preserve rollback history."""
        payload = self._load_payload()
        target_entry = self._get_entry_by_version(payload, model_name=model_name, version=version)
        self._assign_alias(payload, model_name=model_name, alias=CHAMPION_ALIAS, version=version)
        self._save_payload(payload)
        return self._get_entry_by_version(payload, model_name=model_name, version=target_entry.version)

    def rollback_model(self, model_name: str, target_version: str) -> RegistryEntry:
        """Rollback champion traffic to a known-good version."""
        return self.promote_model_version(model_name=model_name, version=target_version)