# Predição com Imputação Avançada

## 📋 Visão Geral

Este projeto implementa um sistema de imputação e predição de dados de séries temporais com comparação entre métodos de stacking e modelos baseline.

## 🚀 Como Começar

Este projeto segue um fluxo de execução em duas etapas:

### **Etapa 1: Imputação (Imputação de Dados)**

O primeiro passo é executar o notebook de imputação:

```
code/imputation/imputation.ipynb
```

Este notebook realiza:
- Carregamento e preparação dos dados originais
- Aplicação de diferentes métodos de imputação (KNNImputer, BackwardFill, ForwardFill, Mean, Median)
- Geração de arquivos imputados na pasta `datasets/imputed-series/`

**Como executar:**
1. Abra o notebook `code/imputation/imputation.ipynb`
2. Execute todas as células do notebook
3. Aguarde a conclusão da imputação dos dados

### **Etapa 2: Treinamento do Modelo de Predição**

Após completar a imputação, siga as instruções detalhadas na pasta:

```
code/prediction/trainer_model/
```

Esta pasta contém:
- Scripts para treinamento de modelos de predição
- Comparação entre método de stacking e modelos baseline
- Exemplos de como executar scripts de treinamento
- Interpretação de resultados

## 📁 Estrutura do Projeto

```
├── code/
│   ├── analysis/              # Scripts de análise
│   ├── imputation/            # Notebook de imputação ⭐ (Execute primeiro!)
│   ├── pre-process-series/    # Pré-processamento
│   └── prediction/            # Scripts de predição
│       └── trainer_model/     # Scripts de treinamento
├── datasets/
│   ├── originals/             # Dados originais
│   ├── imputed-series/        # Dados após imputação (gerados na Etapa 1)
│   └── multivariada-post-process/  # Dados pós-processados
├── results/                   # Resultados finais
└── wgrs-plots/                # Plots
```

## ✅ Lista de Verificação de Execução

- [ ] 1. Execute o notebook: `code/imputation/imputation.ipynb`
- [ ] 2. Verifique se os dados foram gerados em `datasets/imputed-series/`
- [ ] 3. Leia e siga as instruções na pasta `code/prediction/trainer_model/`
- [ ] 4. Execute os scripts de treinamento conforme as instruções
- [ ] 5. Analise os resultados em `results/`

## 📖 Mais Informações

Para detalhes técnicos sobre treinamento de modelos e comparação de resultados, consulte os scripts em:
- [Pasta de Treinamento de Modelos](code/prediction/trainer_model/)

## 📝 Requisitos

Verifique o arquivo `requirements.txt` para todas as dependências necessárias (se existir).