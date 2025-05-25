import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS

mesh = Mesh(jax.devices(), axis_names=('batch',)) # Create a 1D mesh, mapping device axis 'data' to the available devices
def mesh_sharding(*names:str|None)-> NamedSharding:
    return NamedSharding(mesh, PS(*names))


class MeshRules:
    batch = ('batch',)
    replicated = ()
    buffer = (None,'batch') # (None, 'data') means no sharding for the first dimension(Time), and sharding for the second dimension(Batch)


    def call(self,name:str)->NamedSharding:
        return mesh_sharding(getattr(self,name))


mesh_rules = MeshRules(batch=('batch',),replicated=None,buffer=(None,'batch'))