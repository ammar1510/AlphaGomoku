from dataclasses import dataclass
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS


mesh = Mesh(jax.devices(), axis_names=("batch"))


def mesh_sharding(*names: str | None) -> NamedSharding:
    return NamedSharding(mesh, PS(*names))


@dataclass(unsafe_hash=True)
class MeshRules:
    batch: tuple[str | None, ...]
    replicated: tuple[str | None, ...]
    buffer: tuple[str | None, ...]

    def __call__(self, name: str) -> NamedSharding:
        sharding_spec = getattr(self, name)
        return mesh_sharding(*sharding_spec)


mesh_rules = MeshRules(
    batch=("batch",),
    replicated=(),  # Empty tuple for no sharding
    buffer=(None, "batch"),
)
