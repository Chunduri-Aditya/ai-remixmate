"""
scripts/api/task_modules — Domain-specific background task modules.

Each module contains task functions that run in job executor threads.
All task functions share the signature:
    def task_name(job_id: str, **kwargs) -> dict
"""
