# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

# src/topologicpy/ifc/context.py
#
# Drop-in replacement for IfcOpenShell 0.8.x (tested conceptually against 0.8.4-style APIs).
# Goals:
# - Create IfcProject early
# - Assign explicit METRE / SQUARE_METRE / CUBIC_METRE units (no accidental mm defaults)
# - Create a single shared IfcOwnerHistory (optional but preferred), and expose it on ctx
# - Create Model + Body representation contexts
#
# Notes:
# - This file intentionally avoids Blender-dependent API helpers.
# - OwnerHistory creation uses the documented pattern of setting owner.settings.get_user/get_application,
#   but remains resilient if some owner usecases are unavailable.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict

import ifcopenshell
import ifcopenshell.api
from topologicpy.Helper import Helper

# owner settings module path is stable in 0.8.x
try:
    import ifcopenshell.api.owner.settings as owner_settings
except Exception:  # pragma: no cover
    owner_settings = None


@dataclass
class IFCContext:
    project: Any
    owner_history: Optional[Any]
    model_context: Any
    body_context: Any


class IFCContextBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, f: ifcopenshell.file) -> IFCContext:
        # 1) Project (create first)
        project = ifcopenshell.api.run(
            "root.create_entity",
            f,
            ifc_class="IfcProject",
            name=getattr(self.cfg, "project_name", "TopologicPy Project"),
        )

        # 2) Units (explicit metres)
        self._assign_units_metres(f)

        # 3) Owner history (create once, reuse everywhere)
        owner_history = self._ensure_owner_history(f)

        # Assign to the project explicitly (others should be assigned at creation sites)
        self._assign_owner_history(project, owner_history)

        # 4) Representation contexts
        model_context = ifcopenshell.api.run("context.add_context", f, context_type="Model")
        body_context = ifcopenshell.api.run(
            "context.add_context",
            f,
            context_type="Model",
            context_identifier="Body",
            target_view="MODEL_VIEW",
            parent=model_context,
        )

        return IFCContext(
            project=project,
            owner_history=owner_history,
            model_context=model_context,
            body_context=body_context,
        )

    # -------------------------
    # Units
    # -------------------------
    def _assign_units_metres(self, f: ifcopenshell.file) -> None:
        """
        IfcOpenShell 0.8.4 signature:
          assign_unit(file, units=None, length=None, area=None, volume=None)

        We use the dict forms to force METRE and avoid default millimetres.
        """
        try:
            ifcopenshell.api.run(
                "unit.assign_unit",
                f,
                length={"is_metric": True, "raw": "METRE"},
                area={"is_metric": True, "raw": "SQUARE_METRE"},
                volume={"is_metric": True, "raw": "CUBIC_METRE"},
            )
        except Exception as e:
            if not getattr(self.cfg, "silent", False):
                print(f"Warning: unit.assign_unit failed; continuing without explicit unit assignment. ({e})")

    # -------------------------
    # OwnerHistory
    # -------------------------
    def _ensure_owner_history(self, f: ifcopenshell.file) -> Optional[Any]:
        """
        Create ONE consistent IfcOwnerHistory, then disable auto-ownerhistory creation.
        """
        try:
            app = self._ensure_application(f)
            user = self._ensure_user(f)

            if owner_settings is not None:
                # Backup current hooks
                prev_get_app = owner_settings.get_application
                prev_get_user = owner_settings.get_user

                # Enable hooks to allow create_owner_history to succeed
                owner_settings.get_application = lambda _f: app
                owner_settings.get_user = lambda _f: user

            # Create exactly one OwnerHistory
            oh = ifcopenshell.api.run("owner.create_owner_history", f)  # creates a NEW entity each call :contentReference[oaicite:2]{index=2}

            if owner_settings is not None:
                # IMPORTANT: disable hooks so subsequent API entity creation
                # does NOT auto-create additional OwnerHistory entities.
                owner_settings.get_application = lambda _f: None
                owner_settings.get_user = lambda _f: None

                # (Alternative: restore previous hooks if you really need them elsewhere)
                # owner_settings.get_application = prev_get_app
                # owner_settings.get_user = prev_get_user

            return oh

        except Exception as e:
            if not getattr(self.cfg, "silent", False):
                print(f"Warning: owner history not created; continuing without OwnerHistory. ({e})")
            return None

    def _ensure_application(self, f: ifcopenshell.file):
        """
        Create (or retrieve) an IfcApplication authored by TopologicPy/Syntopy.io,
        not IfcOpenShell.

        Ensures:
        IfcApplication.ApplicationDeveloper = IfcOrganization("Syntopy.io"/"TopologicPy")
        """
        app_full_name = getattr(self.cfg, "application_full_name", "TopologicPy")
        app_ident = getattr(self.cfg, "application_identifier", "TopologicPy")
        try:
            app_version = Helper.Version(check=False)
        except Exception:
            app_version = "unknown"

        dev_ident = getattr(self.cfg, "developer_identifier", "SYNTOPY")
        dev_name = getattr(self.cfg, "developer_name", "Syntopy.io")
        dev_desc = getattr(self.cfg, "developer_description", None)

        # 1) Reuse an existing developer organisation if present
        developer_org = None
        try:
            for org in f.by_type("IfcOrganization"):
                if (org.Identification == dev_ident) or (org.Name == dev_name):
                    developer_org = org
                    break
        except Exception:
            developer_org = None

        # 2) Otherwise create it (prefer API helper, fallback to direct entity)
        if developer_org is None:
            try:
                developer_org = ifcopenshell.api.run(
                    "owner.add_organisation",
                    f,
                    identification=dev_ident,
                    name=dev_name,
                )
            except Exception:
                developer_org = f.createIfcOrganization(dev_ident, dev_name, dev_desc, None, None)

        # 3) Try API helper for application (some builds support application_developer kw)
        try:
            return ifcopenshell.api.run(
                "owner.add_application",
                f,
                application_developer=developer_org,
                application_full_name=str(app_full_name),
                application_identifier=str(app_ident),
                version=str(app_version),
            )
        except TypeError:
            # API signature variant (no application_developer kw)
            pass
        except Exception:
            # any other failure → fallback to direct construction
            pass

        # 4) Fallback: create IfcApplication directly (portable and explicit)
        # IFC4: IfcApplication( ApplicationDeveloper, Version, ApplicationFullName, ApplicationIdentifier )
        return f.createIfcApplication(developer_org, str(app_version), str(app_full_name), str(app_ident))

    def _ensure_user(self, f: ifcopenshell.file) -> Any:
        """
        Create a generic Person+Organisation without personal data.
        """
        # Prefer the API helpers (most consistent across 0.8.x)
        try:
            person = ifcopenshell.api.run(
                "owner.add_person",
                f,
                identification="TOPOLOGICPY",
                family_name="TopologicPy",
                given_name="Exporter",
            )
            org = ifcopenshell.api.run(
                "owner.add_organisation",
                f,
                identification="TOPOLOGICPY",
                name="TopologicPy",
            )
            return ifcopenshell.api.run(
                "owner.add_person_and_organisation",
                f,
                person=person,
                organisation=org,
            )
        except Exception:
            # Fallback: create bare minimum IFC entities directly
            person = f.createIfcPerson(None, "Exporter", "TopologicPy", None, None, None, None, None)
            org = f.createIfcOrganization(None, "TopologicPy", None, None, None)
            return f.createIfcPersonAndOrganization(person, org)

    def _assign_owner_history(self, entity: Any, owner_history: Optional[Any]) -> None:
        if owner_history is None or entity is None:
            return
        if hasattr(entity, "OwnerHistory"):
            try:
                entity.OwnerHistory = owner_history
            except Exception:
                if not getattr(self.cfg, "silent", False):
                    print("Warning: Could not assign OwnerHistory to an entity.")