from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSnapshot:
    available: bool
    system_cpu_percent: float | None = None
    system_memory_percent: float | None = None
    system_memory_used_gb: float | None = None
    system_memory_total_gb: float | None = None
    process_cpu_percent: float | None = None
    process_cpu_time_seconds: float | None = None
    process_memory_gb: float | None = None
    memory_pressure_label: str = "unbekannt"
    memory_pressure_percent: float | None = None
    message: str | None = None


def collect_resource_snapshot(process_pid: int | None) -> ResourceSnapshot:
    try:
        import psutil
    except ImportError:
        return ResourceSnapshot(
            available=False,
            message="Ressourcenanzeige braucht `psutil`. Bitte `python -m pip install psutil` ausfuehren.",
        )

    memory = psutil.virtual_memory()
    system_cpu = psutil.cpu_percent(interval=None)
    process_cpu = None
    process_cpu_time = None
    process_memory = None

    if process_pid:
        try:
            root = psutil.Process(process_pid)
            processes = [root] + root.children(recursive=True)
            rss_bytes = 0
            cpu_percent = 0.0
            cpu_seconds = 0.0
            for process in processes:
                try:
                    rss_bytes += process.memory_info().rss
                    cpu_percent += process.cpu_percent(interval=None)
                    times = process.cpu_times()
                    cpu_seconds += times.user + times.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            process_memory = _bytes_to_gb(rss_bytes)
            process_cpu = cpu_percent
            process_cpu_time = cpu_seconds
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_memory = None
            process_cpu = None

    pressure_percent = float(memory.percent)
    return ResourceSnapshot(
        available=True,
        system_cpu_percent=float(system_cpu),
        system_memory_percent=pressure_percent,
        system_memory_used_gb=_bytes_to_gb(memory.used),
        system_memory_total_gb=_bytes_to_gb(memory.total),
        process_cpu_percent=process_cpu,
        process_cpu_time_seconds=process_cpu_time,
        process_memory_gb=process_memory,
        memory_pressure_label=_pressure_label(pressure_percent),
        memory_pressure_percent=pressure_percent,
    )


def _bytes_to_gb(value: int) -> float:
    return value / (1024**3)


def _pressure_label(percent: float) -> str:
    if percent >= 92:
        return "kritisch"
    if percent >= 85:
        return "hoch"
    if percent >= 70:
        return "mittel"
    return "niedrig"
