# Projeto de Reconhecimento de Caracteres

Este projeto implementa um sistema de reconhecimento de caracteres usando técnicas de visão computacional e aprendizado de máquina. Ele foi desenvolvido para escanear e reconhecer caracteres a partir de páginas de livros ou imagens de texto.

## Estrutura do Projeto

- `build_dataset.py`: Cria arquivos de rótulo para imagens de caracteres recortados.
- `build_images.py`: Gera imagens de texto a partir de arquivos de texto de entrada.
- `crop.py`: Recorta caracteres individuais das imagens de texto.
- `inference.py`: Realiza o reconhecimento de caracteres em novas imagens de texto.
- `model.py`: Treina o modelo de reconhecimento de caracteres.
- `preprocess.py`: Gera dados de texto aleatórios para treinamento.

## Configuração

1. Clone o repositório.
2. Instale as dependências necessárias:
   ```
   pip install -r requirements.txt
   ```

## Uso

### Treinando o Modelo

1. Gere dados de treinamento:
   ```
   python preprocess.py
   ```
   Isso criará arquivos de texto no diretório `text`.

2. Pré-processar os dados e treinar o modelo:
   ```
   python model.py
   ```
   Este script irá:
   - Gerar imagens de texto a partir dos arquivos de texto
   - Recortar caracteres individuais
   - Criar arquivos de rótulo
   - Treinar o modelo de reconhecimento de caracteres

O modelo treinado será salvo como `character_recognition_model.h5`, junto com `label_encoder.pkl` e `scaler.pkl`.

### Executando Inferência

1. Coloque seu texto em `./inference/text.txt`.

2. Execute o script de inferência:
   ```
   python inference.py
   ```

Isso gerará uma imagem a partir do seu texto, recortará os caracteres e preverá cada caractere usando o modelo treinado. A string final prevista será impressa no console.

## Arquitetura do Modelo

O modelo de reconhecimento de caracteres utiliza uma rede neural simples com a seguinte arquitetura:
- Camada de entrada
- Camada densa (128 unidades, ativação ReLU)
- Camada densa (64 unidades, ativação ReLU)
- Camada de saída (ativação softmax)

## Aumento de Dados

O processo de treinamento inclui técnicas de aumento de dados, como:
- Adição de ruído Gaussiano
- Deslocamento leve das imagens em diferentes direções

## Desempenho

Após o treinamento, as métricas de desempenho do modelo (precisão e um relatório de classificação detalhado) serão exibidas no console.

## Nota

Este projeto é destinado a fins educacionais e pode necessitar de mais otimizações para uso em produção.

