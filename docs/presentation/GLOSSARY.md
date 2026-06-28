# Glossary — simple terms for speaker notes

| Term | Simple explanation |
|------|-------------------|
| **Evo** | The project name. Evacuation intelligence dashboard + API. |
| **k-NN (k-nearest neighbors)** | Finds the most similar past examples in the dataset and averages their outcomes. No neural net — just distance in feature space. |
| **Feature vector** | A list of numbers describing one site at one moment: occupancy, density, hazard severity, etc. |
| **Hybrid model (Evo 1.2)** | Uses k-NN for some outputs and a neural net for others — whichever is more reliable per metric. |
| **MLP** | Multi-layer perceptron — a standard feed-forward neural network. |
| **LightGBM** | Gradient boosting — a tree-based machine learning model, good at tabular data. |
| **OOF ensemble** | Out-of-fold average: combine MLP and LightGBM predictions from cross-validation without cheating on the test fold. |
| **ONNX** | A portable file format for ML models so the same model runs in Python, cloud, or edge devices. |
| **OpenVINO** | Intel's software to run models fast on CPU or on the Neural Compute Stick USB device. It's the **driver**, not the stick itself. |
| **Neural Compute Stick (NCS)** | Small Intel USB device that accelerates neural network inference on a local computer. |
| **MYRIAD** | OpenVINO's name for the NCS USB device when detected. |
| **R² (R-squared)** | How much better the model is than always guessing the average. 1.0 = perfect; near 0 = weak signal. |
| **MAE** | Mean absolute error — average mistake size (in % or minutes). Lower is better. |
| **DATA_CEILING** | We hit a limit from missing labeled evacuation outcomes, not from lack of model complexity. |
| **Quality gates** | Checklist before promoting a model (error thresholds, beat baselines, export parity, speed). |
| **Cross-validation** | Train on some folds, test on others, repeat — honest estimate of real-world performance. |
| **PeopleSense** | Service that stores occupancy from edge devices (e.g. Raspberry Pi) at schools. |
| **RAG** | Retrieval-augmented generation: pull real facts from our API, then ask an LLM to write text from those facts. |
| **LangChain agents** | Orchestrated GPT steps with tools — used in broadcast mode only. |
| **Leaflet** | Open-source JavaScript map library used in the dashboard. |
| **Vercel** | Hosts the static website; proxies API calls to Oracle. |
| **Oracle VM** | Cloud server running Python FastAPI 24/7. |
| **Neon / SQLite** | Databases storing disaster run history. |
| **Preflight (Evo 1.3)** | Script checks required files exist before training is allowed to start. |
| **Synthetic demo** | Fake drill data clearly labeled — for pipeline demos, not production claims. |
