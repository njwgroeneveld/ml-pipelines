# Postmortem: MLflow 3 on a Raspberry Pi needed three fixes before it stayed up

| | |
|---|---|
| Date | 2026-10-04 |
| Duration | 21 minutes (11:36–11:57 CEST), five Helm revisions |
| Impact | None outside this project: MLflow had no users yet. The shared Supabase project (which also serves live trading) saw no failed logins. |
| Status | Resolved. 543 MB in use, no restarts since. |

## Summary

The MLflow 3.16 tracking server (Helm chart `community-charts/mlflow` 1.11.7) did not stay up on the
8 GB Raspberry Pi 5. The first revision crash-looped on a read-only file system; the next two were
killed for running out of memory, at 1 GiB and at 1.5 GiB. A temporary 3 GiB limit made it possible
to measure where the memory went: MLflow 3 starts a background job runner with eight worker processes
for its GenAI features, about 220 MB each. This project does not use them. With job execution
switched off and two web workers, the server runs in 543 MB under a 1 GiB limit.

## Timeline (CEST)

| Time | Revision | Change | Result |
|---|---|---|---|
| 11:36 | 1 | install, memory limit 1 GiB | crash loop: `OSError: Read-only file system: '/mlflow/metrics'`; the liveness probe keeps restarting the container |
| 11:47 | 2 | `emptyDir` volume on `/mlflow/metrics` | starts, then `OOMKilled` at 1 GiB |
| 11:53 | 3 | limit 1.5 GiB, `--workers=2` | `OOMKilled` again |
| 11:55 | 4 | limit 3 GiB, only to measure | runs; ~1.9 GB in use (cgroup `memory.peak`, RSS per process): most of it a job runner and 8 `huey_consumer` processes |
| 11:57 | 5 | `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`, limit back to 1 GiB | 543 MB, 0 restarts |
| 12:24–12:25 | 6, 7 | release drill: upgrade, then `helm rollback` to 5 | runs and registered models intact: state lives in Postgres and object storage, not in the pod |

## Root causes

1. **Metrics directory on a read-only root file system.** Enabling the chart's ServiceMonitor adds
   `--expose-prometheus=/mlflow/metrics`. MLflow creates that directory at start-up, but the container
   runs with a read-only root file system.
2. **Background job execution is on by default in MLflow 3.** The server starts a job runner plus
   eight `huey` consumers for GenAI background jobs, each loading all of MLflow. The switch is
   `MLFLOW_SERVER_ENABLE_JOB_EXECUTION`, found in `mlflow/environment_variables.py`; the chart does
   not mention it.

Contributing: the memory limit was sized on MLflow 2 experience, and raising it (revision 3) treated
the symptom instead of finding the cause.

## Detection

By hand, with `kubectl get pods` and `kubectl describe pod`. The install script did have an automatic
stop, but it watched only the init containers, the ones that log in to Supabase, because repeated
failed logins make the Supabase pooler block the whole project. It never looked at the main container,
so it saw none of the crashes.

## What went well

- The database login was tested from a throwaway pod before the install, so not a single failed login
  reached Supabase during five revisions.
- Measuring with a temporary high limit (revision 4) showed where the memory went; the fix followed
  in the next revision, two minutes later.
- Every fix is a commented line in `platform/mlflow/values.yaml`, so the next install starts from it.

## What went wrong

- Helm reported revision 1 as "Install complete" while the main container kept restarting: a
  successful install said nothing about whether the server stayed up.
- Revision 3 raised the limit before anyone looked at what used the memory.

## Action items

| Action | State |
|---|---|
| `emptyDir` on `/mlflow/metrics`, job execution off, `--workers=2`, each with a comment in the values | done |
| Release drill: upgrade and `helm rollback` without losing state | done |
| Watch `containerStatuses` (restart count, last termination reason), not only init containers, in install scripts | open |
| Alert on container restarts in namespaces `mlflow` and `argo` (`kube_pod_container_status_restarts_total`) | open, with the planned Grafana dashboard |
