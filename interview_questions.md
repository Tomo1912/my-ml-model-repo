# 10 Potential Questions About the Application and Setup

---

## 1. Why did you choose Random Forest over other algorithms like XGBoost or neural networks?

Random Forest is a solid baseline - robust, requires minimal tuning, and hard to overfit. Perfect choice for a production demo.

---

## 2. Why ONNX format instead of pickle?

Pickle can execute arbitrary Python code when loading - security risk. ONNX is just a mathematical graph definition, no code execution.

---

## 3. Why two separate repos (app repo + GitOps repo)?

Separation of concerns. App repo has code and CI, GitOps repo has only Kubernetes manifests. ArgoCD watches only the GitOps repo - cleaner and more secure.

---

## 4. What happens if ArgoCD goes down?

The application keeps running since it's already deployed. Only new deployments won't happen until ArgoCD is back. You can also manually `kubectl apply`.

---

## 5. How do you handle model versioning?

Image tag is the commit SHA. Model metadata (R², MAE, hash) is stored in `model_metadata.json`. I can track which commit = which model.

---

## 6. What if someone pushes a broken model?

Trivy scans for vulnerabilities but doesn't test model accuracy. We could add a validation step in CI that tests the model on a test dataset before deploying.

---

## 7. How does rate limiting work and why 30 req/min?

SlowAPI middleware tracks IP addresses. 30/min is enough for normal usage while protecting against abuse. In production, we'd use Redis for distributed rate limiting.

---

## 8. Why K3s instead of full Kubernetes or EKS/GKE?

K3s is lightweight (~512MB RAM), perfect for a single-node VPS. EKS/GKE would be overkill and more expensive for this use case.

---

## 9. How do you handle secrets (GitHub token, etc.)?

GitHub Secrets for CI/CD, Kubernetes Secrets for image pull. In production, we'd use External Secrets Operator or HashiCorp Vault.

---

## 10. What would you add for production-ready deployment?

- Horizontal Pod Autoscaler (HPA)
- Prometheus + Grafana monitoring
- Model A/B testing
- Automated model retraining pipeline
- Database for logging predictions

