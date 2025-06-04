# System Architecture

The project exposes a containerised full stack application with three main services:

- **frontend** – Next.js application offering dataset upload, task selection and inference UI.
- **backend** – FastAPI application exposing REST endpoints to manage datasets, start training and run evaluations.
- **model** – optional worker container used for heavy training/inference on GPU hardware.

Services communicate over HTTP and share a mounted volume for datasets and trained models.

```
+-----------+       +------------+       +-------+
|  Browser  | <---> |  frontend  | <---> |backend| <---> model(optional)
+-----------+       +------------+       +-------+
```

The backend mounts `./data` for storing uploaded images/annotations and `./models` for trained checkpoints. The compose file defines the network and dependencies.
