"""appfl[sim]: MPI-backed FL simulation package."""


def run_mpi(config) -> None:
    from appfl_sim.runner import run_mpi as _run_mpi

    _run_mpi(config)


def run_serial(config) -> None:
    from appfl_sim.runner import run_serial as _run_serial

    _run_serial(config)


__all__ = ["run_mpi", "run_serial"]
