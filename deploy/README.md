# Deployment (EKS + Helm)

Everything here is a template. **Nothing in this repository has been deployed to
a cluster** — no AWS account, image registry or Helm release is attached to it.
The manifests are written to be applied, not to claim a live deployment.

## What is here

```
deploy/
└── helm/querymesh/
    ├── Chart.yaml
    ├── values.yaml            # EKS/ALB defaults, HPA, external secrets
    └── templates/
        ├── deployment.yaml    # non-root, probes on /health, env from ConfigMap
        ├── configmap.yaml     # USE_LANGGRAPH, AUTH_ENABLED, TOP_K, ...
        ├── service.yaml
        ├── ingress.yaml       # ALB ingress class, TLS listener
        ├── hpa.yaml           # 2-6 pods on 70% CPU
        ├── serviceaccount.yaml# IRSA annotation slot
        └── externalsecret.yaml# AWS Secrets Manager -> k8s secret
```

Secrets (LLM keys, JWT secret, Redis URL) never live in `values.yaml`: the
ExternalSecret pulls them from AWS Secrets Manager into a Kubernetes secret that
the deployment reads with `envFrom`.

## Prerequisites

```bash
aws eks update-kubeconfig --name <cluster> --region <region>
helm repo add eks https://aws.github.io/eks-charts
# ALB controller, External Secrets Operator and a ClusterSecretStore named
# `aws-secrets-manager` are assumed to exist
```

## Deploy

```bash
# build and push the image (CI does this on main, see .github/workflows/cd.yml)
docker build -t ghcr.io/<owner>/querymesh:0.2.0 .
docker push ghcr.io/<owner>/querymesh:0.2.0

helm upgrade --install querymesh deploy/helm/querymesh \
  --namespace querymesh --create-namespace \
  --set image.tag=0.2.0 \
  --set ingress.host=querymesh.example.com
```

Then check it:

```bash
kubectl -n querymesh get pods,svc,ingress,hpa
kubectl -n querymesh port-forward svc/querymesh-querymesh 8000:80
curl localhost:8000/health
```

## Scaling notes

The API is stateless apart from the in-process index and cache, so it scales
horizontally. Two things to change before real traffic:

- **Index per pod** — every replica builds its own index. Fine for a demo corpus,
  wasteful for a large one: move to a shared vector store and warm the index at
  startup, or run ingestion as a separate CronJob.
- **Cache** — set `REDIS_URL` so cache hits are shared across replicas; the
  in-process dict is per pod.
