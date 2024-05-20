doskey ins=helm --namespace inference-ms install my-inference-ms nemo-ms/nemollm-inference -f "C:\Users\10698046\OneDrive - LTIMindtree\Desktop\AI-LLM\nvidia-nim\triton-backend-custom-values.yaml"
doskey logs=kubectl logs my-inference-ms-0 -n inference-ms -f
doskey ex=kubectl exec -it my-inference-ms-0 -n inference-ms -- /bin/bash
doskey uns=helm --namespace inference-ms uninstall my-inference-ms