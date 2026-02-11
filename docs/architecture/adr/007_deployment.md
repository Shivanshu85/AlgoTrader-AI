# ADR 007: Deployment Platform - Kubernetes vs Docker Swarm vs AWS ECS

**Status:** ACCEPTED  
**Date:** February 2026  
**ADR Number:** 007  

---

## Decision

**SELECTED: Kubernetes**

### Pros
- ✅ Industry standard (Google, AWS, Azure run it)
- ✅ Excellent auto-scaling
- ✅ Self-healing and rolling updates
- ✅ Large ecosystem (Istio, Helm, Prometheus)
- ✅ Multi-cloud support
- ✅ Huge community

### Implementation

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 10
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: stock-predictor-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  type: LoadBalancer
  selector:
    app: api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

**Status:** ✅ ACCEPTED
