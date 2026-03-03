# 🌱 AgroSmart – Protótipo de Reconhecimento de Folhas

Este projeto faz parte do sistema **AgroSmart**, cujo objetivo é aplicar técnicas de visão computacional no agronegócio brasileiro.

Nesta fase, foi desenvolvido um protótipo capaz de classificar imagens de folhas como:

- **Health** → Folha saudável
- **Sick** → Folha com indícios de doença ou praga

O sistema utiliza um modelo treinado no **Teachable Machine** e executa a classificação localmente em Python, exportando os resultados em formato JSON.

---

## 📂 Estrutura do Projeto

```text
idea-seeders-agrosmart/
│
├── input_images/        # Imagens para classificação
├── models/              # Modelo exportado do Teachable
│   ├── keras_model.h5
│   └── labels.txt
│
├── output/              # Resultados gerados
│   └── results.json
│
├── src/
│   └── predict_folder.py
│
├── README.md
└── .gitignore

```
⚙️ Requisitos

- Python 3.11
- TensorFlow 2.15
- NumPy
- Pillow

🚀 Instalação
1) Criar ambiente virtual (Python 3.11)
```
py -3.11 -m venv venv311
```
2) Ativar o ambiente
```
venv311\Scripts\activate
```
3) Instalar dependências
```
pip install -r requirements.txt
```

▶️ Como Executar:

Com o ambiente virtual ativado:
```
python src/predict_folder.py --in_dir input_images --out output/results.json
```