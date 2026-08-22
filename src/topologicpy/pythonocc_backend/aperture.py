from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .topology import Topology


@dataclass(eq=False)
class Aperture(Topology):
    topology: Optional[Topology] = None

    @staticmethod
    def ByTopologyContext(topology, context):
        """
        Creates an aperture represented by the input topology and associates it
        with the host topology referenced by the input context.

        Parameters
        ----------
        topology : Topology
            The topology representing the aperture.
        context : Context
            The context identifying the host topology and its parametric location.

        Returns
        -------
        Aperture
            The created aperture, or None if the operation fails.
        """
        if topology is None or context is None:
            return None

        # Retrieve the host topology from the context.
        host = getattr(context, "topology", None)

        if host is None:
            try:
                host = context.Topology()
            except Exception:
                host = None

        if host is None:
            return None

        # Create the aperture wrapper.
        aperture = Aperture(
            shape=None,
            topology=topology,
        )

        # --------------------------------------------------------------
        # Aperture -> Context
        #
        # Aperture is intentionally shapeless, so this relationship is
        # stored on the aperture wrapper itself.
        # --------------------------------------------------------------
        try:
            aperture.AddContext(context)
        except Exception:
            aperture.contexts = [context]

        # --------------------------------------------------------------
        # Host -> Aperture
        #
        # Persist this against the host's OCCT shape so that the
        # relationship survives reconstruction of the host wrapper.
        # --------------------------------------------------------------
        from .attribute_manager import AttributeManager

        shape = getattr(host, "shape", None)

        valid_shape = shape is not None

        if valid_shape and hasattr(shape, "IsNull"):
            try:
                valid_shape = not shape.IsNull()
            except Exception:
                pass

        if valid_shape:
            manager = AttributeManager.GetInstance()

            if manager.HasApertures(shape):
                apertures = manager.GetApertures(shape)
            else:
                apertures = list(
                    getattr(host, "apertures", []) or []
                )

            apertures.append(aperture)

            manager.SetApertures(
                shape,
                apertures,
            )

            host.apertures = manager.GetApertures(shape)

        else:
            apertures = list(
                getattr(host, "apertures", []) or []
            )

            apertures.append(aperture)
            host.apertures = apertures

        return aperture

    @staticmethod
    def Topology(aperture):
        if not isinstance(aperture, Aperture):
            return None
        return aperture.topology

# ---------------------------------------------------------------------------
# Explicit unsupported Aperture API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _aperture_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Aperture.{name}", return_value)
    return _method


# Aperture.ByTopologyContext is implemented above.
