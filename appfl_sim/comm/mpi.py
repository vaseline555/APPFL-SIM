from __future__ import annotations

from typing import Optional


try:
    from mpi4py import MPI
except Exception:  # pragma: no cover
    MPI = None


class MpiSyncCommunicator:
    """
    APPFL-style synchronous communicator interface for simulator.

    API aligned with APPFL `MpiSyncCommunicator` usage in `run_mpi.py`:
    - gather/scatter
    - broadcast_global_model / recv_global_model_from_server
    - recv_all_local_models_from_clients / send_local_models_to_server
    """

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    def scatter(self, contents, source: int):
        if source == self.rank:
            assert len(contents) == self.size, (
                "The size of contents must equal communicator size in scatter"
            )
        return self.comm.scatter(contents, root=source)

    def gather(self, content, dest: int):
        return self.comm.gather(content, root=dest)

    def broadcast(self, payload, source: int = 0):
        return self.comm.bcast(payload, root=source)

    def barrier(self):
        self.comm.Barrier()

    def broadcast_global_model(self, model=None, args: Optional[dict] = None):
        assert model is not None or args is not None, "Nothing to broadcast"
        if model is None:
            self.comm.bcast((args, False), root=self.rank)
        else:
            self.comm.bcast((args, True), root=self.rank)
            self.comm.bcast(model, root=self.rank)

    def recv_global_model_from_server(self, source: int):
        args, has_model = self.comm.bcast(None, root=source)
        if has_model:
            model = self.comm.bcast(None, root=source)
        else:
            model = None
        return model if args is None else (model, args)

    def recv_all_local_models_from_clients(self):
        """Called by server rank: receive per-rank client payload dictionaries."""
        gathered = self.comm.gather(None, root=self.rank)
        merged = {}
        for r, payload in enumerate(gathered):
            if r == self.rank:
                continue
            if payload:
                merged.update(payload)
        return merged

    def send_local_models_to_server(self, models, dest: int):
        """Called by non-server ranks: send local client payload dictionary."""
        self.comm.gather(models, root=dest)


def get_mpi_comm():
    if MPI is None:
        raise RuntimeError("mpi4py is required.")
    return MPI.COMM_WORLD
