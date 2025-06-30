# README

## Visão Geral do Pipeline
Este projeto implementa um pipeline de aprendizado de máquina para detecção de carcinogênese em microfotografias de fígado de rato, combinando:

- **Extração de características**  
  - Handcrafted: Haralick, LBP, Wavelet etc.  
  - CNN-based: embeddings de modelos pré-treinados (ResNet, EfficientNet etc.)  

- **Pré-processamento**  
  - Variação de scalers: Standard, MinMax, Robust, PowerTransformer  
  - PCA para reduzir dimensão de embeddings  

- **Validação Cruzada Aninhada**  
  - Inner loop com `StratifiedGroupKFold` + `GridSearchCV` (métrica: balanced_accuracy)  
  - Hold-out final no conjunto de teste  

- **Ensembles**  
  - Soft-Voting ponderado (SVM, Logistic Regression, RF, MLP, NB) com calibração  
  - Stacking com meta-classificador GradientBoosting  

- **Visualizações & Avaliações**  
  - Matrizes de confusão  
  - Curvas ROC  
  - UMAP e t-SNE  
  - Heatmaps de distribuição de grupos por fold  

- **Persistência**  
  - JSON com acurácia, melhores parâmetros e relatórios  
  - Figuras salvas em `results/<extrator>/<classificador>/`

---

## Estrutura do Projeto

```
.
├── config.py                    # Definições de MODELS, PCA_COMPONENTS, RESULTS_FILE
├── main.py                      # Script principal
├── utils/
│   ├── data_loader.py           # load_data(), load_or_extract()
│   ├── feature_extractor.py     # extract_cnn_features(), extract_handcrafted_features()
│   ├── evaluator.py             # get_classifiers()
│   ├── logger.py                # save_results()
│   └── save_plots.py            # Funções de plotagem e redução de dimensionalidade
├── cached_features/             # Cache de features handcrafted
├── results/                     # Saída de JSON e figuras
└── README.md                    # Este arquivo
```

---

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Ajuste configurações em `config.py` se necessário (MODELS, PCA_COMPONENTS, RESULTS_FILE).
3. Execute o pipeline:
   ```bash
   python main.py
   ```
4. Confira:
   - **JSON** de resultados (`RESULTS_FILE`)
   - **Figuras** em `results/`

---

## Customizações

- **Modelos CNN**: edite o dicionário `MODELS` em `config.py` para testar modelos infividual ou o handcrafted.  
- **Número de componentes PCA**: ajuste a lista `PCA_COMPONENTS` em `config.py`.  
- **Classificadores e hiperparâmetros**: configure em `utils/evaluator.py`.  
- **Scalers**: modifique `SCALERS` no topo de `main.py`.
- **Handcrafted**: modifique a funcão `extract_handcrafted_features` para testar extratores de features individuais


## Testar End to End pipeline

Copiar o conteunido do archivo `jupyter_notebook_CNN_pipeline.py` num notebook

